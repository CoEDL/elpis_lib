from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from loguru import logger
from transformers import AutoModelForCTC, AutoProcessor, Trainer

from elpis.datasets.processing import create_dataset, prepare_dataset
from elpis.trainer.job import TrainingJob


def train(
    job: TrainingJob,
    output_dir: Path,
    dataset_dir: Path,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Trains a model for use in transcription.

    Parameters:
        job: Info about the training job, e.g. training options.
        output_dir: Where to save the trained model.
        dataset_dir: A directory containing the preprocessed dataset to train with.
        cache_dir: A directory to use for caching HFT downloads and datasets.

    Returns:
        A path to the folder containing the trained model.
    """
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


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch
