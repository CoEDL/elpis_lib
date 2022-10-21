import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from elpis.datasets import CleaningOptions, ProcessingBatch
from elpis.datasets.preprocessing import (
    TARGET_SAMPLE_RATE,
    clean_annotation,
    generate_training_files,
    has_finished_processing,
    process_batch,
)
from elpis.models import Annotation
from elpis.models.elan_options import ElanOptions, ElanTierSelector

ABUI_DATASET_FILES = ["abui_1.eaf", "abui_1.wav", "abui_2.eaf", "abui_2.wav"]
DATA_PATH = Path(__file__).parent.parent / "data"

TEST_ANNOTATION_TIMED = Annotation(
    audio_file=DATA_PATH / "test.wav",
    transcript="hi",
    start_ms=0,
    stop_ms=1000,
)

BATCH = ProcessingBatch(
    transcription_file=DATA_PATH / "abui_4.eaf",
    audio_file=DATA_PATH / "abui_4.wav",
    cleaning_options=CleaningOptions(),
    elan_options=ElanOptions(
        selection_mechanism=ElanTierSelector.NAME, selection_value="Phrase"
    ),
)
TEXT_BATCH = ProcessingBatch(
    transcription_file=DATA_PATH / "oily_rag.txt",
    audio_file=DATA_PATH / "oily_rag.wav",
    cleaning_options=CleaningOptions(),
    elan_options=None,
)


def test_process_elan_batch(tmp_path: Path):
    files = process_batch(BATCH, output_dir=tmp_path)
    files = list(files)
    assert len(files) == 4


def test_process_text_batch(tmp_path: Path):
    files = process_batch(TEXT_BATCH, output_dir=tmp_path)
    files = list(files)
    assert len(files) == 2


@pytest.fixture()
def untimed_annotation(tmp_path: Path) -> Annotation:
    return Annotation(audio_file=tmp_path / "test.wav", transcript="hi")


@pytest.fixture()
def timed_annotation(tmp_path: Path) -> Annotation:
    return Annotation(
        audio_file=tmp_path / "test.wav",
        transcript="hi",
        start_ms=0,
        stop_ms=1000,
    )


@pytest.fixture()
def cleaner(mocker) -> Mock:
    return mocker.patch("elpis.datasets.preprocessing.clean_text")


@pytest.fixture()
def cutter(mocker) -> Mock:
    return mocker.patch("elpis.datasets.preprocessing.audio.cut")


@pytest.fixture()
def resampler(mocker) -> Mock:
    return mocker.patch("elpis.datasets.preprocessing.audio.resample")


@pytest.fixture()
def copier(mocker) -> Mock:
    return mocker.patch("elpis.datasets.preprocessing.shutil.copy")


def test_clean_annotation(cleaner: Mock, untimed_annotation: Annotation):
    cleaner.return_value = "wow"

    result = clean_annotation(untimed_annotation, CleaningOptions())
    cleaner.assert_called_once()
    assert result.transcript == "wow"
    assert untimed_annotation.transcript == "hi"


def test_generate_training_files_with_untimed_annotation(
    tmp_path: Path,
    copier: Mock,
    cutter: Mock,
    resampler: Mock,
    untimed_annotation: Annotation,
):
    audio_file = tmp_path / "test.wav"

    transcription, audio = generate_training_files(
        untimed_annotation, output_dir=tmp_path
    )
    cutter.assert_not_called()
    copier.assert_not_called()
    resampler.assert_called_once_with(
        audio_path=audio_file, destination=audio_file, sample_rate=TARGET_SAMPLE_RATE
    )

    assert transcription == tmp_path / "test.json"
    assert audio == audio_file

    with open(transcription) as f:
        annotation = Annotation.from_dict(json.load(f))
        assert annotation.audio_file == audio
        assert annotation.transcript == untimed_annotation.transcript


def test_generate_training_files_with_untimed_annotation_and_copy(
    tmp_path: Path,
    cutter: Mock,
    copier: Mock,
    resampler: Mock,
    untimed_annotation: Annotation,
):
    audio_file = tmp_path / "test.wav"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_audio_file = output_dir / audio_file.name

    transcription, audio = generate_training_files(untimed_annotation, output_dir)
    cutter.assert_not_called()
    copier.assert_called_once()
    resampler.assert_called_once_with(
        audio_path=output_audio_file,
        destination=output_audio_file,
        sample_rate=TARGET_SAMPLE_RATE,
    )

    assert transcription == output_dir / "test.json"
    assert audio.name == output_audio_file.name

    with open(transcription) as f:
        annotation = Annotation.from_dict(json.load(f))
        assert annotation.audio_file == audio
        assert annotation.transcript == untimed_annotation.transcript


def test_generate_training_files_with_timed_annotation(
    tmp_path: Path,
    cutter: Mock,
    copier: Mock,
    resampler: Mock,
    timed_annotation: Annotation,
):
    transcription, audio = generate_training_files(timed_annotation, tmp_path)
    cutter.assert_called_once()
    copier.assert_not_called()
    resampler.assert_called_once()
    assert transcription == tmp_path / f"test_{timed_annotation.start_ms}.json"
    assert audio == tmp_path / f"test_{timed_annotation.start_ms}.wav"

    with open(transcription) as f:
        annotation = Annotation.from_dict(json.load(f))
        assert annotation.audio_file == audio
        assert annotation.transcript == timed_annotation.transcript


def test_has_finished_processing():
    processed_files = ["abui_1.json", "abui_1.wav", "abui_2.json", "abui_2.wav"]
    assert has_finished_processing(ABUI_DATASET_FILES, processed_files)


def test_has_finished_procesing_with_path_prefixes():
    processed_files = ["abui_1.json", "abui_1.wav", "abui_2.json", "abui_2.wav"]
    prefix = "someUserId/datasetName/"
    processed_files = [prefix + name for name in processed_files]
    assert has_finished_processing(ABUI_DATASET_FILES, processed_files)


def test_has_finished_processing_with_incomplete_files_should_return_false():
    processed_files = ["abui_1.json", "abui_1.wav"]
    assert not has_finished_processing(ABUI_DATASET_FILES, processed_files)


def test_has_finished_processing_with_timed_annotations():
    processed_files = [
        "abui_1.json",
        "abui_1.wav",
        "abui_2_1000.json",
        "abui_2_1000.wav",
    ]
    assert has_finished_processing(ABUI_DATASET_FILES, processed_files)


def test_has_finished_processing_with_timed_and_multiple_annotations():
    processed_files = [
        "abui_1.json",
        "abui_1.wav",
        "abui_2_1000.json",
        "abui_2_1000.wav",
        "abui_2_3000.json",
        "abui_2_3000.wav",
    ]
    assert has_finished_processing(ABUI_DATASET_FILES, processed_files)
