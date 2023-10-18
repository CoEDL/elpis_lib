from pathlib import Path

import pytest
from transformers import TrainingArguments

from elpis.models.job import DataArguments, Job, ModelArguments


@pytest.fixture
def job(tmp_path: Path):
    model_dir = tmp_path / "model"

    return Job(
        model_args=ModelArguments(
            "facebook/wav2vec2-base",
        ),
        data_args=DataArguments(
            dataset_name_or_path="mozilla-foundation/common_voice_11_0",
            dataset_config_name="gn",
        ),
        training_args=TrainingArguments(output_dir=str(model_dir)),
    )


def test_save_job(tmp_path: Path, job: Job):
    file = tmp_path / "job.json"
    job.save(file)

    assert file.is_file()
    assert Job.from_json(file) == job


def test_job_serialization(job: Job):
    data = job.to_dict()
    assert Job.from_dict(data) == job
