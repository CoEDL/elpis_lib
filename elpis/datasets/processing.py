import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Audio, DatasetDict, load_dataset
from loguru import logger
from transformers import AutoFeatureExtractor, AutoTokenizer

from elpis.models.job import Job

LOGGING_TRANSCRIPT_SAMPLE = 2


def create_dataset(
    job: Job,
    test_size: float = 0.2,
) -> DatasetDict:
    """Creates a dataset with test/train splits from the data within a given
    directory.

    Parameters:
        job: The training job to run.
        test_size: The percentage of the dataset to allocate as the test set.

    Returns:
        A dataset dictionary with test and train splits.
    """
    dataset_path = Path(job.data_args.dataset_name_or_path)
    if not dataset_path.is_dir():
        raise ValueError(
            f"Attempting to create local dataset from non-existent "
            f"directory: {dataset_path}."
        )

    transcript_files = [
        str(dataset_path / file)
        for file in os.listdir(dataset_path)
        if (dataset_path / file).suffix == ".json"
    ]
    logger.debug(
        f"Transcript file paths sample: {transcript_files[:LOGGING_TRANSCRIPT_SAMPLE]}"
    )

    dataset = load_dataset("json", data_files=transcript_files, cache_dir=job.model_args.cache_dir)  # type: ignore

    # Convert the audio file name column into the matching audio data
    audio_column = job.data_args.audio_column_name
    dataset = dataset.rename_column("audio_file", audio_column)
    logger.debug(f"Dataset audio file paths sample: {dataset['train'][audio_column][:LOGGING_TRANSCRIPT_SAMPLE]}")  # type: ignore

    def resolve_audio_path(row: Dict[str, Any]) -> Dict[str, Any]:
        # Forcefully resolve to same dir as dataset.
        path = dataset_path / Path(row[audio_column]).name
        row[audio_column] = str(path.absolute())
        return row

    dataset = dataset.map(resolve_audio_path)
    logger.debug(f"Dataset audio file paths post-resolution: {dataset['train'][audio_column][:LOGGING_TRANSCRIPT_SAMPLE]}")  # type: ignore

    dataset = dataset["train"].train_test_split(test_size=test_size, seed=job.training_args.seed)  # type: ignore
    # rename test to eval
    dataset["eval"] = dataset["test"]
    dataset.pop("test")

    return dataset


def prepare_dataset(
    job: Job,
    tokenizer: AutoTokenizer,
    feature_extractor: AutoFeatureExtractor,
    dataset: DatasetDict,
) -> DatasetDict:
    """Runs some preprocessing over the given dataset.

    Parameters:
        dataset: The dataset on which to apply the preprocessing
        processor: The processor to apply over the dataset
    """

    # Load the audio data and resample if necessary.
    dataset = dataset.cast_column(
        job.data_args.audio_column_name,
        Audio(sampling_rate=feature_extractor.sampling_rate),  # type: ignore
    )

    def _prepare_dataset(batch: Dict) -> Dict[str, List]:
        audio = batch[job.data_args.audio_column_name]
        inputs = feature_extractor(  # type: ignore
            audio["array"], sampling_rate=audio["sampling_rate"]
        )

        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}
        phoneme_language = job.data_args.phoneme_language
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch[job.data_args.text_column_name], **additional_kwargs).input_ids  # type: ignore
        return batch

    max_input_length = (
        job.data_args.max_duration_in_seconds * feature_extractor.sampling_rate  # type: ignore
    )
    min_input_length = (
        job.data_args.min_duration_in_seconds * feature_extractor.sampling_rate  # type: ignore
    )

    def is_audio_in_length_range(length: int):
        return length >= min_input_length and length <= max_input_length

    with job.training_args.main_process_first(desc="dataset map preprocessing"):
        worker_count = job.data_args.preprocessing_num_workers
        dataset = dataset.map(
            _prepare_dataset,
            remove_columns=next(iter(dataset.values())).column_names,
            num_proc=worker_count,
            desc="preprocess datasets",
        )

        # filter data that is shorter than min_input_length
        dataset = dataset.filter(
            is_audio_in_length_range,
            num_proc=worker_count,
            input_columns=["input_length"],
        )

    return dataset
