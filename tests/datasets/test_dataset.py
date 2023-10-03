from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import List, Optional

import pytest

from elpis.datasets import CleaningOptions, Dataset, ProcessingBatch
from elpis.models import ElanOptions
from elpis.models.elan_options import ElanTierSelector

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
class Files(Enum):
    ELAN = ["1.eaf", "1.wav"]
    TEXT = ["1.txt", "1.wav"]
    MISMATCHED = ["1.eaf", "1.wav", "2.wav", "3.txt"]
    COLLIDING = ["1.eaf", "1.wav", "1.txt"]
    MESSY = ["1.eaf", "1.wav", "2.eaf", "2.txt", "2.wav", "3.eaf", "4.wav"]


def create_dataset(files: Files, elan_options: Optional[ElanOptions] = None) -> Dataset:
    paths = [Path(x) for x in files.value]
    print(f"Creating Dataset for: {files}: with {paths}")
    return Dataset(
        name="dataset",
        files=paths,
        cleaning_options=CleaningOptions.from_dict(CLEANING_OPTIONS_DICT),
        elan_options=elan_options,
    )


@pytest.fixture
def elan_options():
    return ElanOptions.from_dict(ELAN_OPTIONS_DICT)


@pytest.fixture
def dataset():
    return create_dataset(Files.ELAN)


@pytest.fixture
def text_dataset():
    return create_dataset(Files.TEXT)


@pytest.fixture
def elan_dataset(elan_options):
    return create_dataset(Files.ELAN, elan_options=elan_options)


@pytest.fixture
def mismatched_dataset(elan_options):
    return create_dataset(Files.MISMATCHED, elan_options=elan_options)


@pytest.fixture
def colliding_dataset(elan_options):
    return create_dataset(Files.COLLIDING, elan_options=elan_options)


@pytest.fixture
def messy_dataset(elan_options):
    return create_dataset(Files.MESSY, elan_options=elan_options)


DATASET_DICT = {
    "name": "dataset",
    "files": Files.ELAN.value,
    "cleaning_options": CLEANING_OPTIONS_DICT,
}
MESSY_DATASET_DICT = {
    "name": "dataset",
    "files": Files.MESSY.value,
    "cleaning_options": CLEANING_OPTIONS_DICT,
}

DATASET_DICT_ELAN = DATASET_DICT | {"elan_options": ELAN_OPTIONS_DICT}


def to_paths(names: List[str]) -> List[Path]:
    return [Path(name) for name in names]

def test_build_dataset():
    dataset = Dataset.from_dict(DATASET_DICT)
    assert dataset.name == "dataset"
    assert dataset.files == to_paths(Files.ELAN.value)
    assert dataset.cleaning_options == CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert dataset.elan_options is None


def test_build_dataset_with_elan():
    dataset = Dataset.from_dict(DATASET_DICT_ELAN)
    assert dataset.name == "dataset"
    assert dataset.files == to_paths(Files.ELAN.value)
    assert dataset.cleaning_options == CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert dataset.elan_options == ElanOptions.from_dict(ELAN_OPTIONS_DICT)


def test_serialize_dataset(dataset: Dataset, elan_dataset: Dataset):
    assert dataset.to_dict() == DATASET_DICT
    assert elan_dataset.to_dict() == DATASET_DICT_ELAN


def test_dataset_is_valid(dataset: Dataset, messy_dataset: Dataset):
    assert dataset.is_valid()
    assert not messy_dataset.is_valid()


def test_dataset_is_empty(dataset: Dataset):
    assert not dataset.is_empty()

    dataset.files = []
    assert dataset.is_empty()
    assert not dataset.is_valid()


def test_dataset_has_elan(elan_dataset: Dataset, text_dataset: Dataset):
    assert elan_dataset.has_elan()
    assert not text_dataset.has_elan()


def test_dataset_mismatched_files(dataset: Dataset, mismatched_dataset: Dataset):
    assert len(dataset.mismatched_files) == 0
    assert mismatched_dataset.mismatched_files == {Path("2.wav"), Path("3.txt")}


def test_duplicate_files(dataset: Dataset, colliding_dataset: Dataset):
    assert len(dataset.colliding_files) == 0
    assert colliding_dataset.colliding_files == {Path("1.eaf"), Path("1.txt")}


def test_valid_transcriptions(messy_dataset: Dataset):
    assert len(list(messy_dataset.valid_transcriptions)) == 1


def test_dataset_batching(dataset: Dataset):
    batches = list(dataset.to_batches())
    assert len(batches) == 1
    job = batches[0]
    transcript_file, audio_file = to_paths(Files.ELAN.value)
    assert job.transcription_file == transcript_file
    assert job.audio_file == audio_file
    assert job.cleaning_options == dataset.cleaning_options
    assert job.elan_options == dataset.elan_options


# ====== Processing Job ======
VALID_BATCH_DICT = {
    "transcription_file": Files.ELAN.value[0],
    "audio_file": Files.ELAN.value[1],
    "cleaning_options": CLEANING_OPTIONS_DICT,
    "elan_options": ELAN_OPTIONS_DICT,
}


def test_build_processing_job():
    job = ProcessingBatch.from_dict(VALID_BATCH_DICT)
    transcript_file, audio_file = to_paths(Files.ELAN.value)
    assert job.transcription_file == transcript_file
    assert job.audio_file == audio_file
    assert job.cleaning_options == CleaningOptions.from_dict(CLEANING_OPTIONS_DICT)
    assert job.elan_options == ElanOptions.from_dict(ELAN_OPTIONS_DICT)


def test_serialize_processing_job():
    job = ProcessingBatch.from_dict(VALID_BATCH_DICT)
    assert job.to_dict() == VALID_BATCH_DICT
