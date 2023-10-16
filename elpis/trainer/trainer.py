import warnings
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
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from elpis.datasets import create_dataset, prepare_dataset
from elpis.models.job import Job
from elpis.models.vocab import VOCAB_FILE, Vocab
from elpis.trainer.data_collator import DataCollatorCTCWithPadding
from elpis.trainer.metrics import create_metrics
from elpis.trainer.utils import log_to_file


def run_job(
    job: Job,
    log_file: Optional[Path] = None,
) -> Path:
    """Fine-tunes a model for use in transcription.

    Parameters:
        job: Info about the training job, e.g. training options.
        dataset_dir: A directory containing the preprocessed dataset to train with.
        log_file: An optional file to write training logs to.

    Returns:
        A path to the folder containing the trained model.
    """

    logging_context = log_to_file(log_file) if log_file is not None else nullcontext()
    with logging_context:
        # Setup required directories.
        output_dir = job.training_args.output_dir
        cache_dir = job.model_args.cache_dir
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        set_seed(job.training_args.seed)

        logger.info("Preparing Datasets...")
        config = create_config(job)
        dataset = create_dataset(job)

        tokenizer = create_tokenizer(job, config, dataset)
        logger.info(f"Tokenizer: {tokenizer}")  # type: ignore
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            job.model_args.model_name_or_path,
            cache_dir=cache_dir,
            token=job.data_args.token,
            trust_remote_code=job.data_args.trust_remote_code,
        )
        dataset = prepare_dataset(job, tokenizer, feature_extractor, dataset)
        logger.info("Finished Preparing Datasets")

        update_config(job, config, tokenizer)

        logger.info("Downloading pretrained model...")
        model = create_ctc_model(job, config)
        logger.info("Downloaded model.")

        # Now save everything to be able to create a single processor later
        # make sure all processes wait until data is saved
        logger.info("Saving config, tokenizer and feature extractor.")
        with job.training_args.main_process_first():
            # only the main process saves them
            if is_main_process(job.training_args.local_rank):
                feature_extractor.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)  # type: ignore
                config.save_pretrained(output_dir)  # type: ignore

        try:
            processor = AutoProcessor.from_pretrained(job.training_args.output_dir)
        except (OSError, KeyError):
            warnings.warn(
                "Loading a processor from a feature extractor config that does not"
                " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
                " attribute to your `preprocessor_config.json` file to suppress this warning: "
                " `'processor_class': 'Wav2Vec2Processor'`",
                FutureWarning,
            )
            processor = Wav2Vec2Processor.from_pretrained(job.training_args.output_dir)

        data_collator = DataCollatorCTCWithPadding(processor=processor)  # type: ignore

        # Initialize Trainer
        trainer = Trainer(
            model=model,  # type: ignore
            data_collator=data_collator,
            args=job.training_args,
            compute_metrics=create_metrics(job.data_args.eval_metrics, processor),
            train_dataset=dataset["train"] if job.training_args.do_train else None,  # type: ignore
            eval_dataset=dataset["eval"] if job.training_args.do_eval else None,  # type: ignore
            tokenizer=processor,  # type: ignore
        )

        logger.info(f"Begin training model...")
        train(job, trainer, dataset)
        logger.info(f"Finished training!")

        evaluate(job, trainer, dataset)
        clean_up(job, trainer)

        return Path(output_dir)


def create_config(job: Job) -> AutoConfig:
    return AutoConfig.from_pretrained(
        job.model_args.model_name_or_path,
        cache_dir=job.model_args.cache_dir,
        token=job.data_args.token,
        trust_remote_code=job.data_args.trust_remote_code,
    )


def create_tokenizer(
    job: Job, config: AutoConfig, dataset: DatasetDict
) -> AutoTokenizer:
    tokenizer_name_or_path = job.model_args.tokenizer_name_or_path
    if tokenizer_name_or_path is not None:
        return AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            token=job.data_args.token,
            trust_remote_code=job.data_args.trust_remote_code,
        )

    training_args = job.training_args

    # save vocab in training output dir
    tokenizer_name_or_path = job.training_args.output_dir
    vocab_file = Path(tokenizer_name_or_path) / VOCAB_FILE

    # Delete existing vocab file if overwriting
    with training_args.main_process_first():
        if training_args.overwrite_output_dir and vocab_file.is_file():
            try:
                vocab_file.unlink()
            except OSError:
                # in shared file-systems it might be the case that
                # two processes try to delete the vocab file at the some time
                pass

    # Build up a vocab from the dataset.
    with training_args.main_process_first(desc="dataset map vocabulary creation"):
        if not vocab_file.is_file():
            Path(tokenizer_name_or_path).mkdir(exist_ok=True, parents=True)
            text_column = job.data_args.text_column_name

            vocab = Vocab.from_strings(dataset["train"][text_column])
            if "test" in dataset:
                test_vocab = Vocab.from_strings(dataset["test"][text_column])
                vocab = vocab.merge(test_vocab)

            vocab.add(job.data_args.unk_token)
            vocab.add(job.data_args.pad_token)
            vocab.replace(" ", job.data_args.word_delimiter_token)
            logger.info(f"Vocab: {vocab.vocab}")
            vocab.save(vocab_file)

    # If the tokenizer has just been created,
    # it is defined by `tokenizer_class` if present in config else by `model_type`
    tokenizer_kwargs = {
        "config": config if config.tokenizer_class is not None else None,  # type: ignore
        "tokenizer_type": config.model_type if config.tokenizer_class is None else None,  # type: ignore
        "unk_token": job.data_args.unk_token,
        "pad_token": job.data_args.pad_token,
        "word_delimiter_token": job.data_args.word_delimiter_token,
        "do_lower_case": job.data_args.do_lower_case,
    }

    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        token=job.data_args.token,
        trust_remote_code=job.data_args.trust_remote_code,
        **tokenizer_kwargs,
    )


def update_config(job: Job, config: AutoConfig, tokenizer: AutoTokenizer) -> None:
    config.update(  # type: ignore
        {
            "feat_proj_dropout": job.model_args.feat_proj_dropout,
            "attention_dropout": job.model_args.attention_dropout,
            "hidden_dropout": job.model_args.hidden_dropout,
            "final_dropout": job.model_args.final_dropout,
            "mask_time_prob": job.model_args.mask_time_prob,
            "mask_time_length": job.model_args.mask_time_length,
            "mask_feature_prob": job.model_args.mask_feature_prob,
            "mask_feature_length": job.model_args.mask_feature_length,
            "gradient_checkpointing": job.training_args.gradient_checkpointing,
            "layerdrop": job.model_args.layerdrop,
            "ctc_loss_reduction": job.model_args.ctc_loss_reduction,
            "ctc_zero_infinity": job.model_args.ctc_zero_infinity,
            "pad_token_id": tokenizer.pad_token_id,  # type: ignore
            "bos_token_id": tokenizer.bos_token_id,  # type: ignore
            "eos_token_id": tokenizer.eos_token_id,  # type: ignore
            "vocab_size": len(tokenizer),  # type: ignore
            "activation_dropout": job.model_args.activation_dropout,
        }
    )


def create_ctc_model(job: Job, config: AutoConfig) -> AutoModelForCTC:
    model = AutoModelForCTC.from_pretrained(
        job.model_args.model_name_or_path,
        cache_dir=job.model_args.cache_dir,
        config=config,
        token=job.data_args.token,
        trust_remote_code=job.data_args.trust_remote_code,
    )

    # freeze encoder
    if job.model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    return model


def last_checkpoint(job: Job) -> Optional[str]:
    """Returns the string corresponding to the path or name of the last
    training checkpoint, if it exists."""
    training_args = job.training_args
    output_dir = Path(training_args.output_dir)

    if not output_dir.is_dir():
        return None
    if not training_args.do_train:
        return None
    if training_args.overwrite_output_dir:
        return None

    checkpoint = get_last_checkpoint(training_args.output_dir)
    checkpoint_folders = [path for path in output_dir.iterdir() if path.is_dir()]

    if checkpoint is None and len(checkpoint_folders) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Set `overwrite_output_dir` in training_args to overcome."
        )
    elif checkpoint is not None:
        logger.info(
            f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

    return checkpoint


def train(job: Job, trainer: Trainer, dataset: DatasetDict):
    if not job.training_args.do_train:
        logger.info("Skipping training: `job.training_args.do_train` is false.")
        return

    checkpoint = last_checkpoint(job)
    if checkpoint is None and Path(job.model_args.model_name_or_path).is_dir():
        checkpoint = job.model_args.model_name_or_path

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    metrics = train_result.metrics

    # Add training samples to metrics
    max_train_samples = (
        job.data_args.max_train_samples
        if job.data_args.max_train_samples is not None
        else len(dataset["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(dataset["train"]))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model()
    trainer.save_state()


def evaluate(job: Job, trainer: Trainer, dataset: DatasetDict):
    if not job.training_args.do_eval:
        logger.info("Skipping eval: `job.training_args.do_eval` is false.")
        return

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    max_eval_samples = (
        job.data_args.max_eval_samples
        if job.data_args.max_eval_samples is not None
        else len(dataset["eval"])
    )
    metrics["eval_samples"] = min(max_eval_samples, len(dataset["eval"]))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    logger.info(metrics)


def clean_up(job: Job, trainer: Trainer):
    """Writes a model card, and optionally pushes the trained model to the
    huggingface hub."""
    config_name = (
        job.data_args.dataset_config_name
        if job.data_args.dataset_config_name is not None
        else "na"
    )
    kwargs = {
        "finetuned_from": job.model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": ["automatic-speech-recognition", job.data_args.dataset_name_or_path],
        "dataset_args": (
            f"Config: {config_name}, Training split: {job.data_args.train_split_name}, Eval split:"
            f" {job.data_args.eval_split_name}"
        ),
        "dataset": f"{job.data_args.dataset_name_or_path.upper()} - {config_name.upper()}",
    }
    if "common_voice" in job.data_args.dataset_name_or_path:
        kwargs["language"] = config_name

    if job.training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
