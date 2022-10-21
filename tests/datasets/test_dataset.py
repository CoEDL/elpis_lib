from pathlib import Path
from typing import List

from elpis.datasets import CleaningOptions, Dataset, ProcessingBatch
from elpis.models import ElanOptions

# ====== Elan Options ======
ELAN_OPTIONS_DICT = {
    "selection_mechanism": "tier_name",
    "selection_value": "test",
}
INVALID_ELAN_OPTIONS_DICT = {
    "selection_mechanism": "pier_name",
    "selection_value": "jest",
}


# ====== Dataset Options ======
CLEANING_OPTIONS_DICT = {
    "punctuation_to_remove": ":",
    "punctuation_to_explode": ";",
    "words_to_remove": ["<UNK>"],
}


def test_default_dataset_options():
    options = CleaningOptions()
    assert options.punctuation_to_remove == ""
    assert options.punctuation_to_explode == ""
    assert options.words_to_remove == []


def test_build_dataset_options():
    options = CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert options.punctuation_to_remove == ":"
    assert options.punctuation_to_explode == ";"
    assert options.words_to_remove == ["<UNK>"]


def test_serialize_dataset_options():
    options = CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert options.to_dict() == CLEANING_OPTIONS_DICT


# ====== Dataset ======
FILES_WITH_ELAN = ["1.eaf", "1.wav"]
FILES_WITHOUT_ELAN = ["1.txt", "1.wav"]
MISMATCHED_FILES = ["1.eaf", "1.wav", "2.wav", "3.txt"]
COLLIDING_FILES = ["1.eaf", "1.wav", "1.txt"]


DATASET_DICT = {
    "name": "dataset",
    "files": FILES_WITH_ELAN,
    "cleaning_options": CLEANING_OPTIONS_DICT,
}

DATASET_DICT_ELAN = DATASET_DICT | {"elan_options": ELAN_OPTIONS_DICT}


def to_paths(names: List[str]) -> List[Path]:
    return [Path(name) for name in names]


def test_build_dataset():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert dataset.name == "dataset"
    assert dataset.files == to_paths(FILES_WITH_ELAN)
    assert dataset.cleaning_options == CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert dataset.elan_options is None


def test_build_dataset_with_elan():
    dataset = Dataset.from_dict(DATASET_DICT_ELAN)
    assert dataset.name == "dataset"
    assert dataset.files == to_paths(FILES_WITH_ELAN)
    assert dataset.cleaning_options == CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert dataset.elan_options == ElanOptions.from_dict(ELAN_OPTIONS_DICT)


def test_serialize_dataset():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert dataset.to_dict() == DATASET_DICT

    dataset = Dataset.from_dict(DATASET_DICT_ELAN)
    assert dataset.to_dict() == DATASET_DICT_ELAN


def test_dataset_is_valid():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert dataset.is_valid()


def test_dataset_is_empty():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert not dataset.is_empty()

    dataset.files = []
    assert dataset.is_empty()
    assert not dataset.is_valid()


def test_dataset_has_elan():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert dataset.has_elan()

    dataset.files = to_paths(FILES_WITHOUT_ELAN)
    assert not dataset.has_elan()


def test_dataset_mismatched_files():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert len(dataset.mismatched_files()) == 0

    dataset.files = to_paths(MISMATCHED_FILES)
    assert set(dataset.mismatched_files()) == {Path("2.wav"), Path("3.txt")}


def test_duplicate_files():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert len(dataset.colliding_files()) == 0

    dataset.files = to_paths(COLLIDING_FILES)
    assert set(dataset.colliding_files()) == {Path("1.eaf"), Path("1.txt")}


def test_dataset_batching():
    dataset = Dataset.from_dict(DATASET_DICT)
    batch = dataset.to_batches()
    assert len(batch) == 1
    job = batch[0]
    transcript_file, audio_file = to_paths(FILES_WITH_ELAN)
    assert job.transcription_file == transcript_file
    assert job.audio_file == audio_file
    assert job.cleaning_options == dataset.cleaning_options
    assert job.elan_options == dataset.elan_options


# ====== Processing Job ======
VALID_BATCH_DICT = {
    "transcription_file": FILES_WITH_ELAN[0],
    "audio_file": FILES_WITH_ELAN[1],
    "cleaning_options": CLEANING_OPTIONS_DICT,
    "elan_options": ELAN_OPTIONS_DICT,
}


def test_build_processing_job():
    job = ProcessingBatch.from_dict(VALID_BATCH_DICT)
    transcript_file, audio_file = to_paths(FILES_WITH_ELAN)
    assert job.transcription_file == transcript_file
    assert job.audio_file == audio_file
    assert job.cleaning_options == CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert job.elan_options == ElanOptions.from_dict(ELAN_OPTIONS_DICT)


def test_serialize_processing_job():
    job = ProcessingBatch.from_dict(VALID_BATCH_DICT)
    assert job.to_dict() == VALID_BATCH_DICT
