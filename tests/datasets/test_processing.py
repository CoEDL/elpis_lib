import os
import shutil
from pathlib import Path

from loguru import logger
from transformers import TrainingArguments

from elpis.datasets.processing import create_local_dataset
from elpis.models.job import DataArguments, Job, ModelArguments

DATA_PATH = Path(__file__).parent.parent / "data" / "processing"


def test_create_local_dataset(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "out"

    for directory in cache_dir, dataset_dir, model_dir, output_dir:
        directory.mkdir(exist_ok=True, parents=True)

    logger.info(DATA_PATH)
    for file in os.listdir(DATA_PATH):
        if Path(file).suffix in [".wav", ".json"]:
            shutil.copy(DATA_PATH / file, dataset_dir)

    logger.info(os.listdir(dataset_dir))

    job = Job(
        model_args=ModelArguments(
            model_name_or_path="facebook/wav2vec2-base", cache_dir=str(cache_dir)
        ),
        data_args=DataArguments(
            dataset_name_or_path=str(dataset_dir), text_column_name="transcript"
        ),
        training_args=TrainingArguments(
            output_dir=str(model_dir),
        ),
    )

    dataset = create_local_dataset(job)
    assert "train" in dataset
    assert "eval" in dataset
