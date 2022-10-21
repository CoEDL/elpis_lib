import os
import shutil
from pathlib import Path

from loguru import logger

from elpis.datasets.processing import create_dataset

DATA_PATH = Path(__file__).parent.parent / "data" / "processing"


def test_create_dataset(tmp_path: Path):
    logger.info(DATA_PATH)
    for file in os.listdir(DATA_PATH):
        if Path(file).suffix in [".wav", ".json"]:
            shutil.copy(DATA_PATH / file, tmp_path)

    logger.info(os.listdir(tmp_path))

    result = create_dataset(tmp_path, tmp_path / "cache")
    assert "test" in result
    assert "train" in result
