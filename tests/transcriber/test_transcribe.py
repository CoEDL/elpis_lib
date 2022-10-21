from pathlib import Path

from loguru import logger

from elpis.trainer.job import BASE_MODEL
from elpis.transcriber.transcribe import build_pipeline, transcribe

DATA_DIR = Path(__file__).parent.parent / "data"
AUDIO = DATA_DIR / "oily_rag.wav"


def test_standalone_inference(tmp_path: Path):
    pipeline = build_pipeline(BASE_MODEL, tmp_path)
    preds = pipeline(str(AUDIO), return_timestamps="word")
    logger.info(preds)
