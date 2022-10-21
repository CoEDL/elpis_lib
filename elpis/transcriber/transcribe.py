from pathlib import Path
from typing import Any, Dict, List

from transformers import AutomaticSpeechRecognitionPipeline as ASRPipeline
from transformers import AutoModelForCTC, AutoProcessor, pipeline

from elpis.models import Annotation

CACHE_DIR = Path("/tmp")
TASK = "automatic-speech-recognition"


def build_pipeline(
    pretrained_location: str, cache_dir: Path = CACHE_DIR
) -> ASRPipeline:
    """Builds the pipeline from the supplied pretrained location.

    Parameters:
        pretrained_location: A huggingface model name, or local path to the
            pretrained model.
        cache_dir: The directory in which to store temporary files.

    Returns:
        A pipeline to be used for asr.
    """
    model = AutoModelForCTC.from_pretrained(pretrained_location, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(pretrained_location, cache_dir=cache_dir)

    return pipeline(
        task=TASK,
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )  # type: ignore


def transcribe(audio: Path, asr: ASRPipeline) -> List[Annotation]:
    """Transcribes the given audio and gives back the resulting annotations.

    Parameters:
        audio: The path to the audio file to transcribe.
        asr: The automatic speech recognition pipeline.

    Returns:
        A list of the inferred annotations in the given audio.
    """
    preds: Dict[str, Any] = asr(str(audio), return_timestamps="word")  # type: ignore
    chunks = preds["chunks"]

    return list(map(lambda chunk: annotation_from_chunk(chunk, audio), chunks))


def annotation_from_chunk(chunk: Dict[str, Any], audio_file: Path) -> Annotation:
    """Builds an annotation from a chunk.

    Chunks are in the form:
        {"text": "some_text", "timestamp": (start, stop)}

    Parameters:
        chunk: The chunk to convert
        audio_file: The file which the chunk is for.

    Returns:
        The corresponding annotation.
    """
    text = chunk["text"]
    start, stop = chunk["timestamps"]

    return Annotation(
        transcript=text,
        start_ms=start * 1000,
        stop_ms=stop * 1000,
        audio_file=audio_file,
    )
