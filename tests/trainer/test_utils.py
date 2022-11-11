from pathlib import Path

from loguru import logger

from elpis.trainer.utils import log_to_file


def test_log_to_file_captures_loguru_logs(tmp_path: Path):
    log_file = tmp_path / "logs.txt"

    @log_to_file(log_file)
    def test_function():
        logger.info("TEST")

    test_function()
    assert log_file.exists()
    with open(log_file) as _logs:
        assert "TEST" in _logs.read()


def test_log_to_file_with_missing_file_does_nothing():
    @log_to_file(None)
    def test_function():
        return 2

    result = test_function()
    assert result == 2
