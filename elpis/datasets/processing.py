import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Audio, DatasetDict, load_dataset
from transformers import Wav2Vec2Processor

PROCESSOR_COUNT = 4
AUDIO_COLUMN = "audio"
SAMPLING_RATE = 16_000


def create_dataset(
    dataset_path: Path, cache_dir: Optional[Path] = None, test_size: float = 0.2
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
    # Annoying hack
    if cache_dir is not None:
        cache_dir = str(cache_dir)  # type: ignore

    dataset = load_dataset("json", data_files=transcript_files, cache_dir=cache_dir)  # type: ignore

    # Convert the audio file name column into the matching audio data
    dataset = dataset.rename_column("audio_file", AUDIO_COLUMN)

    def resolve_audio_path(row: Dict[str, Any]) -> Dict[str, Any]:
        row[AUDIO_COLUMN] = str(dataset_path / row[AUDIO_COLUMN])
        return row

    dataset = dataset.map(resolve_audio_path)
    dataset = dataset.cast_column(AUDIO_COLUMN, Audio(sampling_rate=SAMPLING_RATE))

    return dataset["train"].train_test_split(test_size=test_size)  # type: ignore


def prepare_dataset(dataset: DatasetDict, processor: Wav2Vec2Processor) -> DatasetDict:
    """Runs some preprocessing over the given dataset.

    TODO: I'm going to be honest, I have no idea what this does, and need some
    smart ML knight in shining armour to write a propert description.

    Parameters:
        dataset: The dataset to apply the preprocessing
        processor: The processor to apply over the dataset
    """

    def prepare_dataset(batch: Dict) -> Dict[str, List]:
        audio = batch["audio"]

        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcript"]).input_ids

        return batch

    return dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=PROCESSOR_COUNT,
    )
