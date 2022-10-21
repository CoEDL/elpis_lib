import json
from pathlib import Path
from unittest.mock import Mock

from elpis.datasets import CleaningOptions, ProcessingBatch
from elpis.datasets.preprocessing import (
    clean_annotation,
    generate_training_files,
    has_finished_processing,
    process_batch,
)
from elpis.models import Annotation
from elpis.models.elan_options import ElanOptions, ElanTierSelector

TEST_ANNOTATION = Annotation(audio_file=Path("test.wav"), transcript="hi")
TEST_ANNOTATION_TIMED = Annotation(
    audio_file=Path("test.wav"),
    transcript="hi",
    start_ms=0,
    stop_ms=1000,
)


ABUI_DATASET_FILES = ["abui_1.eaf", "abui_1.wav", "abui_2.eaf", "abui_2.wav"]
DATA_PATH = Path(__file__).parent.parent / "data"

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


def test_clean_annotation(mocker):
    cleaner_mock: Mock = mocker.patch("elpis.datasets.preprocessing.clean_text")
    cleaner_mock.return_value = "wow"

    result = clean_annotation(TEST_ANNOTATION, CleaningOptions())
    cleaner_mock.assert_called_once()
    assert result.transcript == "wow"
    assert TEST_ANNOTATION.transcript == "hi"


def test_generate_training_files_with_untimed_annotation(tmp_path: Path, mocker):
    cut_mock: Mock = mocker.patch("elpis.datasets.preprocessing.audio.cut")
    copy_mock: Mock = mocker.patch("elpis.datasets.preprocessing.shutil.copy")

    audio_file = tmp_path / "test.wav"
    annotation = Annotation(audio_file=audio_file, transcript="hey")

    transcription, audio = generate_training_files(annotation, output_dir=tmp_path)
    cut_mock.assert_not_called()
    copy_mock.assert_not_called()

    assert transcription == tmp_path / "test.json"
    assert audio == audio_file

    with open(transcription) as f:
        assert Annotation.from_dict(json.load(f)) == annotation


def test_generate_training_files_with_untimed_annotation_and_copy(
    tmp_path: Path, mocker
):
    cut_mock: Mock = mocker.patch("elpis.datasets.preprocessing.audio.cut")
    copy_mock: Mock = mocker.patch("elpis.datasets.preprocessing.shutil.copy")

    audio_file = tmp_path / "test.wav"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    transcription, audio = generate_training_files(TEST_ANNOTATION, output_dir)
    cut_mock.assert_not_called()
    copy_mock.assert_called_once()

    assert transcription == output_dir / "test.json"
    assert audio.name == (output_dir / audio_file.name).name

    with open(transcription) as f:
        assert Annotation.from_dict(json.load(f)) == TEST_ANNOTATION


def test_generate_training_files_with_timed_annotation(tmp_path: Path, mocker):
    cut_mock: Mock = mocker.patch("elpis.datasets.preprocessing.audio.cut")
    copy_mock: Mock = mocker.patch("elpis.datasets.preprocessing.shutil.copy")

    transcription, audio = generate_training_files(TEST_ANNOTATION_TIMED, tmp_path)
    cut_mock.assert_called_once()
    copy_mock.assert_not_called()
    assert transcription == tmp_path / f"test_{TEST_ANNOTATION_TIMED.start_ms}.json"
    assert audio == tmp_path / f"test_{TEST_ANNOTATION_TIMED.start_ms}.wav"

    with open(transcription) as f:
        assert Annotation.from_dict(json.load(f)) == TEST_ANNOTATION_TIMED


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
