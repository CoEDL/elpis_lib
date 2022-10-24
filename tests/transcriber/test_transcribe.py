from pathlib import Path

from loguru import logger
from pytest import mark

from elpis.trainer.job import BASE_MODEL
from elpis.transcriber.transcribe import build_pipeline, transcribe

DATA_DIR = Path(__file__).parent.parent / "data"
AUDIO = DATA_DIR / "oily_rag.wav"


@mark.integration
def test_standalone_inference():
    pipeline = build_pipeline(BASE_MODEL)
    preds = pipeline(str(AUDIO), return_timestamps="word")
    logger.info(preds)


@mark.integration
def test_transcribe():
    pipeline = build_pipeline(BASE_MODEL)
    annotations = transcribe(AUDIO, pipeline)
    assert len(annotations) > 0
