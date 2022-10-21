from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import TrainingArguments

BASE_MODEL = "facebook/wav2vec2-base-960h"
SAMPLING_RATE = 16_000


class TrainingStatus(Enum):
    WAITING = "waiting"
    TRAINING = "training"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class TrainingOptions:
    batch_size: int = 4
    epochs: int = 2
    learning_rate: float = 1e-4
    min_duration: int = 0
    max_duration: int = 60
    word_delimiter_token: str = " "
    test_size: float = 0.2

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingOptions":
        field_names = [field.name for field in fields(TrainingOptions)]
        kwargs = {key: data[key] for key in data if key in field_names}
        return TrainingOptions(**kwargs)


@dataclass
class TrainingJob:
    model_name: str
    dataset_name: str
    options: TrainingOptions
    status: TrainingStatus = TrainingStatus.WAITING
    base_model: str = BASE_MODEL
    sampling_rate: int = SAMPLING_RATE

    def to_training_args(self, output_dir: Path) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(output_dir),
            group_by_length=True,
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=self.options.epochs,
            fp16=True if torch.cuda.is_available() else False,
            gradient_checkpointing=True,
            learning_rate=self.options.learning_rate,
            weight_decay=0.005,
            save_total_limit=2,
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> TrainingJob:
        return TrainingJob(
            model_name=data["model_name"],
            dataset_name=data["dataset_name"],
            options=TrainingOptions.from_dict(data["options"]),
            status=TrainingStatus(data["status"]),
            base_model=data.get("base_model", BASE_MODEL),
            sampling_rate=data.get("sampling_rate", SAMPLING_RATE),
        )
