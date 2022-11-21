from pathlib import Path

from loguru import logger

from elpis.trainer.utils import log_to_file


def test_log_to_file_captures_loguru_logs(tmp_path: Path):
    log_file = tmp_path / "logs.txt"

    with log_to_file(log_file):
        logger.info("TEST")

    assert log_file.exists()
    with open(log_file) as _logs:
        assert "TEST" in _logs.read()
