from elpis.trainer import TrainingJob, TrainingOptions, TrainingStatus


def test_training_options_serialization_round_trip():
    expected = TrainingOptions()
    assert expected == TrainingOptions.from_dict(expected.to_dict())


def test_training_job_serialization_round_trip():
    expected = TrainingJob("model", "dataset", TrainingOptions())
    assert expected == TrainingJob.from_dict(expected.to_dict())
