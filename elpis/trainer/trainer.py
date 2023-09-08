from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
from tokenizers import Tokenizer
from transformers import AutoModelForCTC, AutoProcessor, EvalPrediction, Trainer

from elpis.datasets import create_dataset, prepare_dataset
from elpis.trainer.data_collator import DataCollatorCTCWithPadding
from elpis.trainer.job import TrainingJob
from elpis.trainer.metrics import create_metrics
from elpis.trainer.utils import log_to_file


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
        processor = AutoProcessor.from_pretrained(job.base_model, cache_dir=cache_dir)
        dataset = prepare_dataset(dataset, processor)
        logger.info("Finished Preparing Datasets")

        logger.info("Downloading pretrained model...")
        model = AutoModelForCTC.from_pretrained(
            job.base_model,
            cache_dir=cache_dir,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            # From https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
            # attention_dropout=0.0,
            # hidden_dropout=0.0,
            # feat_proj_dropout=0.0,
            # mask_time_prob=0.05,
            # layerdrop=0.0,
            # vocab_size=len(processor.tokenizer),
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
            tokenizer=processor.feature_extractor,
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
