import sys
from pathlib import Path

from loguru import logger
from transformers.utils import logging

from elpis.trainer.utils import log_to_file


def test_log_to_file_captures_loguru_logs(tmp_path: Path):
    log_file = tmp_path / "logs.txt"

    with log_to_file(log_file):
        logger.info("TEST")

    assert log_file.exists(), "Couldn't find log file"
    with open(log_file) as _logs:
        assert "TEST" in _logs.read()


def test_log_to_file_captures_hugging_face_logs(tmp_path: Path):
    log_file = tmp_path / "logs.txt"

    with log_to_file(log_file):
        logger = logging.get_logger("transformers")
        logger.info("TEST")

    assert log_file.exists(), "Couldn't find log file"
    with open(log_file) as _logs:
        assert "TEST" in _logs.read()


def test_log_to_file_captures_hugging_face_tqdm_logs(tmp_path: Path):
    log_file = tmp_path / "logs.txt"

    original_stderr = sys.stderr
    with log_to_file(log_file):
        for _ in logging.tqdm(range(3)):
            pass

    assert original_stderr is sys.stderr, "Original stderr not replaced!"
    assert log_file.exists(), "Couldn't find log file"
    with open(log_file) as _logs:
        assert len(_logs.read()) > 0, "Didn't capture tqdm logs"
