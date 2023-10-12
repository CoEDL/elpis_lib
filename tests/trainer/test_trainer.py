import os
import shutil
from pathlib import Path

from pytest import mark

from elpis.trainer import TrainingJob, TrainingOptions, train

DATA_PATH = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_PATH / "processing"

JOB = TrainingJob(
    model_name="test",
    dataset_name="test",
    options=TrainingOptions(epochs=1, max_duration=10),
)


@mark.integration
def test_training_succeeds(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True, parents=True)

    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True, parents=True)

    log_file = tmp_path / "logs.txt"

    for file in os.listdir(DATASET_PATH):
        shutil.copy(DATASET_PATH / file, dataset_dir)

    model_path = train(
        job=JOB, output_dir=output_dir, dataset_dir=dataset_dir, log_file=log_file
    )
    assert model_path == output_dir
