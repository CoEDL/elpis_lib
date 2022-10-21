from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import torch
from loguru import logger
from trainer.dataset import create_dataset, prepare_dataset
from trainer.model_metadata import TrainingJob
from transformers import AutoModelForCTC, AutoProcessor, Trainer


def train(metadata: TrainingJob, data_path: Path, dataset_path: Path) -> Path:
    """Trains a model for use in transcription.

    Saves the model to {data_path}/output.

    Parameters:
        metadata: Metadata about the training job, e.g. training options.
        data_path: A directory to use for temporary working files and models.
        dataset_path: A directory containing the dataset to train with.

    Returns:
        A path to the folder containing the trained model.
    """
    cache_dir = data_path / "cache"

    logger.info("Preparing Datasets...")
    dataset = create_dataset(metadata, dataset_path, cache_dir)
    processor = AutoProcessor.from_pretrained(metadata.base_model, cache_dir=cache_dir)
    dataset = prepare_dataset(dataset, processor)
    logger.info("Finished Preparing Datasets")

    logger.info("Downloading pretrained model...")
    model = AutoModelForCTC.from_pretrained(
        metadata.base_model,
        cache_dir=cache_dir,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    logger.info("Downloaded model.")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    output_path = data_path / "output"
    output_path.mkdir(exist_ok=True, parents=True)

    trainer = Trainer(
        model=model,
        args=metadata.to_training_args(output_path),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
    )

    logger.info(f"Begin training model...")
    trainer.train()
    logger.info(f"Finished training!")

    logger.info(f"Saving model @ {output_path}")
    trainer.save_model()
    trainer.save_state()
    logger.info(f"Model written to disk.")
    return output_path


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
