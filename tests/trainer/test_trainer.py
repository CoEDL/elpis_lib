import os
import shutil
from pathlib import Path

from pytest import mark
from transformers import TrainingArguments

from elpis.models.job import DataArguments, Job, ModelArguments
from elpis.trainer import run_job

DATA_PATH = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_PATH / "processing"


@mark.integration
def test_training_succeeds(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "out"

    log_file = tmp_path / "logs.txt"

    for directory in dataset_dir, model_dir, output_dir:
        directory.mkdir(exist_ok=True, parents=True)

    for file in os.listdir(DATASET_PATH):
        shutil.copy(DATASET_PATH / file, dataset_dir)

    job = Job(
        model_args=ModelArguments(model_name_or_path="facebook/wav2vec2-base"),
        data_args=DataArguments(
            dataset_name_or_path=str(dataset_dir), text_column_name="transcript"
        ),
        training_args=TrainingArguments(
            output_dir=str(model_dir),
            num_train_epochs=2,
            learning_rate=1e-4,
            do_train=True,
        ),
    )

    model_path = run_job(job, log_file)
    assert model_path == model_dir
