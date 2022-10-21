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
def test_training(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True, parents=True)

    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True, parents=True)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    for file in os.listdir(DATASET_PATH):
        shutil.copy(DATASET_PATH / file, dataset_dir)

    model_path = train(JOB, output_dir, dataset_dir, cache_dir)
    assert model_path == output_dir
