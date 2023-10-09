import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import sounddevice as sd
from datasets import Audio, DatasetDict, load_dataset
from loguru import logger
from transformers import Wav2Vec2Processor

PROCESSOR_COUNT = 4
AUDIO_COLUMN = "audio"
SAMPLING_RATE = 16_000
LOGGING_TRANSCRIPT_SAMPLE = 2


def create_dataset(
    dataset_path: Path,
    cache_dir: Optional[Path] = None,
    test_size: float = 0.2,
) -> DatasetDict:
    """Creates a dataset with test/train splits from the data within a given
    directory.

    Parameters:
        dataset_path: The path to the unprocessed dataset files.
        cache_dir: The path to save the processed dataset.
        test_size: The percentage of the dataset to allocate as the test set.

    Returns:
        A dataset dictionary with test and train splits.
    """
    transcript_files = [
        str(dataset_path / file)
        for file in os.listdir(dataset_path)
        if (dataset_path / file).suffix == ".json"
    ]
    logger.debug(
        f"Transcript file paths sample: {transcript_files[:LOGGING_TRANSCRIPT_SAMPLE]}"
    )

    # Annoying hack
    if cache_dir is not None:
        cache_dir = str(cache_dir)  # type: ignore

    dataset = load_dataset("json", data_files=transcript_files, cache_dir=cache_dir)  # type: ignore

    # Convert the audio file name column into the matching audio data
    dataset = dataset.rename_column("audio_file", AUDIO_COLUMN)
    logger.debug(f"Dataset audio file paths sample: {dataset['train'][AUDIO_COLUMN][:LOGGING_TRANSCRIPT_SAMPLE]}")  # type: ignore

    def resolve_audio_path(row: Dict[str, Any]) -> Dict[str, Any]:
        # Forcefully resolve to same dir as dataset.
        path = dataset_path / Path(row[AUDIO_COLUMN]).name
        row[AUDIO_COLUMN] = str(path.absolute())
        return row

    dataset = dataset.map(resolve_audio_path)
    logger.debug(f"Dataset audio file paths post-resolution: {dataset['train'][AUDIO_COLUMN][:LOGGING_TRANSCRIPT_SAMPLE]}")  # type: ignore

    def load_audio(batch: Dict) -> Dict:
        path = batch[AUDIO_COLUMN]
        data, sr = librosa.load(path, sr=SAMPLING_RATE, mono=True)

        batch["audio"] = {"path": path, "array": data, "sampling_rate": SAMPLING_RATE}
        return batch

    # dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=SAMPLING_RATE))
    dataset = dataset.map(load_audio)

    # logger.debug(f"Sample audio col values: {dataset['train'][AUDIO_COLUMN][0]}")  # type: ignore
    return dataset["train"].train_test_split(test_size=test_size)  # type: ignore


def prepare_dataset(dataset: DatasetDict, processor: Wav2Vec2Processor) -> DatasetDict:
    """Runs some preprocessing over the given dataset.

    Parameters:
        dataset: The dataset on which to apply the preprocessing
        processor: The processor to apply over the dataset
    """
    logger.debug(f"Dataset pre-prep: {dataset}")
    logger.debug(f"Transcript sample: {dataset['train']['transcript'][0]}")
    logger.debug(
        f'Input array shape:, {np.asarray(dataset["train"][0]["audio"]["array"]).shape}'
    )
    logger.debug(f'Sampling rate:, {dataset["train"][0]["audio"]["sampling_rate"]}')

    def _prepare_dataset(batch: Dict) -> Dict[str, List]:
        # Also from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
        audio = batch[AUDIO_COLUMN]

        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = processor(text=batch["transcript"]).input_ids

        return batch

    columns = dataset.column_names.values()
    # flatten and make unique between datasets
    columns_to_remove = list(set(chain.from_iterable(columns)))

    dataset = dataset.map(
        _prepare_dataset,
        remove_columns=columns_to_remove,
        num_proc=PROCESSOR_COUNT,
    )
    logger.debug(f"Dataset post prep: {dataset}")
    logger.debug(f"Training labels: {dataset['train']['labels'][0]}")

    return dataset
