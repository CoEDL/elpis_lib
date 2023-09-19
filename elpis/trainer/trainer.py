from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from datasets import DatasetDict
from loguru import logger
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from elpis.datasets import create_dataset, prepare_dataset
from elpis.models.vocab import VOCAB_FILE, Vocab
from elpis.trainer.data_collator import DataCollatorCTCWithPadding
from elpis.trainer.job import TrainingJob
from elpis.trainer.metrics import create_metrics
from elpis.trainer.utils import log_to_file


def create_processor(
    job: TrainingJob,
    output_dir: Path,
    dataset: DatasetDict,
    cache_dir: Optional[Path],
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
) -> Wav2Vec2Processor:
    config = AutoConfig.from_pretrained(job.base_model)
    tokenizer_type = config.model_type if config.tokenizer_class is None else None
    config = config if config.tokenizer_class is not None else None

    # Build up a vocab from the dataset.
    train_vocab = Vocab.from_strings(dataset["train"]["transcript"])
    test_vocab = Vocab.from_strings(dataset["test"]["transcript"])
    vocab = train_vocab.merge(test_vocab)

    vocab.add(unk_token)
    vocab.add(pad_token)
    vocab.replace(" ", word_delimiter_token)  # feels a little restrictive?
    logger.info(f"Vocab: {vocab.vocab}")
    vocab.save(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        config=config,
        tokenizer_type=tokenizer_type,
        unk_token=unk_token,
        pad_token=pad_token,
        word_delimiter_token=word_delimiter_token,
        cache_dir=cache_dir,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        job.base_model, cache_dir=cache_dir
    )

    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def train(
    job: TrainingJob,
    output_dir: Path,
    dataset_dir: Path,
    cache_dir: Optional[Path] = None,
    log_file: Optional[Path] = None,
) -> Path:
    """Fine-tunes a model for use in transcription.

    Parameters:
        job: Info about the training job, e.g. training options.
        output_dir: Where to save the trained model.
        dataset_dir: A directory containing the preprocessed dataset to train with.
        cache_dir: A directory to use for caching HFT downloads and datasets.
        log_file: An optional file to write training logs to.

    Returns:
        A path to the folder containing the trained model.
    """

    context = log_to_file(log_file) if log_file is not None else nullcontext()
    with context:
        logger.info("Preparing Datasets...")
        dataset = create_dataset(dataset_dir, cache_dir)
        processor = create_processor(job, output_dir, dataset, cache_dir)
        dataset = prepare_dataset(dataset, processor)
        logger.info("Finished Preparing Datasets")

        logger.info("Downloading pretrained model...")
        model = AutoModelForCTC.from_pretrained(
            job.base_model,
            cache_dir=cache_dir,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,  # type: ignore
            # Wav2vec2 specific hyperparams copied from docs.
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            vocab_size=len(processor.tokenizer),  # type: ignore
            # For Ash -> errors if below param not set.
            ignore_mismatched_sizes=True,
        )
        logger.info("Downloaded model.")

        if job.options.freeze_feature_extractor:
            model.freeze_feature_extractor()

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        output_dir.mkdir(exist_ok=True, parents=True)

        trainer = Trainer(
            model=model,
            args=job.to_training_args(output_dir),
            train_dataset=dataset["train"],  # type: ignore
            eval_dataset=dataset["test"],  # type: ignore
            tokenizer=processor.feature_extractor,  # type: ignore
            data_collator=data_collator,
            compute_metrics=create_metrics(job.metrics, processor),
        )

        logger.info(f"Begin training model...")
        trainer.train()
        logger.info(f"Finished training!")

        logger.info(f"Saving model @ {output_dir}")
        trainer.save_model()
        trainer.save_state()
        processor.save_pretrained(output_dir)
        logger.info(f"Model written to disk.")

        metrics = trainer.evaluate()
        logger.info("==== Metrics ====")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info(metrics)

        return output_dir
