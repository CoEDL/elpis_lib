from elpis.trainer import TrainingJob, TrainingOptions, TrainingStatus
from elpis.trainer.job import BASE_MODEL, SAMPLING_RATE


def test_training_options_serialization_round_trip():
    expected = TrainingOptions()
    assert expected == TrainingOptions.from_dict(expected.to_dict())


def test_training_job_serialization_round_trip():
    expected = TrainingJob("model", "dataset", TrainingOptions())
    assert expected == TrainingJob.from_dict(expected.to_dict())


def test_job_from_basic_dict():
    data = dict(model_name="a", dataset_name="b", options=TrainingOptions().to_dict())
    job = TrainingJob.from_dict(data)
    assert job.model_name == "a"
    assert job.dataset_name == "b"
    assert job.options == TrainingOptions()
    assert job.status == TrainingStatus.WAITING
    assert job.sampling_rate == SAMPLING_RATE
    assert job.base_model == BASE_MODEL
