from pathlib import Path
from typing import Optional

from loguru import logger
from transformers import AutoModelForCTC, AutoProcessor, Trainer

from elpis.datasets import create_dataset, prepare_dataset
from elpis.trainer.data_collator import DataCollatorCTCWithPadding
from elpis.trainer.job import TrainingJob
from elpis.trainer.utils import log_to_file


def train(
    job: TrainingJob,
    output_dir: Path,
    dataset_dir: Path,
    cache_dir: Optional[Path] = None,
    log_file: Optional[Path] = None,
) -> Path:
    """Trains a model for use in transcription.

    Parameters:
        job: Info about the training job, e.g. training options.
        output_dir: Where to save the trained model.
        dataset_dir: A directory containing the preprocessed dataset to train with.
        cache_dir: A directory to use for caching HFT downloads and datasets.
        log_file: An optional file to write training logs to.

    Returns:
        A path to the folder containing the trained model.
    """

    @log_to_file(log_file)
    def _train():
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
        )

        logger.info(f"Begin training model...")
        trainer.train()
        logger.info(f"Finished training!")

        logger.info(f"Saving model @ {output_dir}")
        trainer.save_model()
        trainer.save_state()
        processor.save_pretrained(output_dir)
        logger.info(f"Model written to disk.")

        return output_dir

    return _train()
