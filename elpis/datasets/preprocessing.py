import json
import shutil
from copy import copy
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple

from loguru import logger

import elpis.utils.audio as audio
from elpis.datasets.clean_text import clean_text
from elpis.datasets.dataset import CleaningOptions, ProcessingBatch
from elpis.datasets.extract_annotations import extract_annotations
from elpis.models.annotation import Annotation

DEFAULT_DIR = Path("/tmp")
TARGET_SAMPLE_RATE = 16_000


def process_batch(
    batch: ProcessingBatch, output_dir: Path = DEFAULT_DIR
) -> Iterable[Path]:
    """Generates training files from the processing batch and puts them in
    the given directory.

    Parameters:
        batch: The processing batch to generate files from
        output_dir: The directory in which to stick the files.

    Returns:
        The paths of the generated files.
    """
    annotations = extract_annotations(
        transcription_file=batch.transcription_file, elan_options=batch.elan_options
    )

    annotations = map(
        lambda annotation: clean_annotation(annotation, batch.cleaning_options),
        annotations,
    )

    # Generate training files from the annotations
    return chain(
        *map(
            lambda annotation: generate_training_files(
                annotation, output_dir=output_dir
            ),
            annotations,
        )
    )


def clean_annotation(
    annotation: Annotation, cleaning_options: CleaningOptions
) -> Annotation:
    """Cleans the text within an annotation.

    Parameters:
        annotation: The annotation to clean.
        cleaning_options: The cleaning options for the dataset.

    Returns:
        A new annotation whose transcript has been cleaned.
    """
    transcript = clean_text(
        text=annotation.transcript,
        words_to_remove=cleaning_options.words_to_remove,
        punctuation_to_explode=cleaning_options.punctuation_to_explode,
        punctuation_to_remove=cleaning_options.punctuation_to_remove,
    )
    result = copy(annotation)
    result.transcript = transcript
    return result


def generate_training_files(
    annotation: Annotation, output_dir: Path = DEFAULT_DIR
) -> Tuple[Path, Path]:
    """Generates a transcript and audio file pairing for this annotation.

    If the annotation is timed (has a start and stop time), we return a path
    to a new audio file, which is constrained to the given times. Otherwise,
    the annotation spans the entire audio path, and so we return this path,
    unmodified.

    Parameters:
        annotation: The annotation for a given section of audio within the
            supplied audio_file.
        output_dir: The directory in which to store the generated files.

    Returns:
        A tuple containing a transcription and audio file path for the given
            annotation.
    """
    # Get a unique name prefix based on annotation start time
    audio_file = annotation.audio_file
    name = audio_file.stem
    if annotation.start_ms is not None:
        name = f"{name}_{annotation.start_ms}"

    # Save audio file.
    if annotation.is_timed():
        cut_audio_file = output_dir / f"{name}.wav"
        audio.cut(
            audio_path=audio_file,
            destination=cut_audio_file,
            start_ms=annotation.start_ms,  # type: ignore
            stop_ms=annotation.stop_ms,  # type: ignore
        )
        audio_file = cut_audio_file
    else:
        # Make sure we're putting the audio file in the output dir
        if audio_file.parent != output_dir:
            shutil.copy(str(audio_file), str(output_dir / audio_file.name))
            audio_file = output_dir / audio_file.name

    # Resample audio to standardise for training
    audio.resample(
        audio_path=audio_file,
        destination=audio_file,
        sample_rate=TARGET_SAMPLE_RATE,
    )

    # Save gimped transcription_file
    next_annotation = Annotation(
        audio_file=audio_file, transcript=annotation.transcript
    )
    transcription_file = output_dir / f"{name}.json"
    with open(transcription_file, "w") as f:
        json.dump(next_annotation.to_dict(), f)

    return transcription_file, audio_file


def has_finished_processing(
    dataset_files: List[str], processed_files: List[str]
) -> bool:
    """Checks whether the dataset has finished processing.

    Parameters:
        dataset_files: A list of names of the files in the dataset.
        processed_files: A list of names of files uploaded to cloud storage for
            the corresponding dataset.

    Returns:
        true iff the supplied list of processed files would be a valid
        processed dataset for the initial files.
    """

    required_stems = {Path(name).stem for name in dataset_files}
    uploaded_stems = {Path(name).stem for name in processed_files}

    def is_processed(required_stem: str) -> bool:
        starts_with_required_stem = lambda stem: stem.startswith(required_stem)
        return any(map(starts_with_required_stem, uploaded_stems))

    return all(map(is_processed, required_stems))
