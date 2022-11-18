from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutomaticSpeechRecognitionPipeline as ASRPipeline
from transformers import AutoModelForCTC, AutoProcessor, pipeline

from elpis.models import Annotation

TASK = "automatic-speech-recognition"


def build_pipeline(
    pretrained_location: str,
    processor_location: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> ASRPipeline:
    """Builds the pipeline from the supplied pretrained location.

    Parameters:
        pretrained_location: A huggingface model name, or local path to the
            pretrained model.
        cache_dir: The directory in which to store temporary files.

    Returns:
        A pipeline to be used for asr.
    """
    if processor_location is None:
        processor_location = pretrained_location

    processor = AutoProcessor.from_pretrained(processor_location, cache_dir=cache_dir)
    model = AutoModelForCTC.from_pretrained(
        pretrained_location,
        cache_dir=cache_dir,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    return pipeline(
        task=TASK,
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )  # type: ignore


def transcribe(audio: Path, asr: ASRPipeline, chunk_length_s=10) -> List[Annotation]:
    """Transcribes the given audio and gives back the resulting annotations.

    Parameters:
        audio: The path to the audio file to transcribe.
        asr: The automatic speech recognition pipeline.
        chunk_length_s: The amount of seconds per audio chunk in the pipeline.

    Returns:
        A list of the inferred annotations in the given audio.
    """
    preds: Dict[str, Any] = asr(str(audio), chunk_length_s=chunk_length_s, return_timestamps="word")  # type: ignore
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
    start, stop = chunk["timestamp"]

    return Annotation(
        transcript=text,
        start_ms=int(start * 1000),
        stop_ms=int(stop * 1000) + 1,
        audio_file=audio_file,
    )
