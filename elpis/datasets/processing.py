import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Audio, DatasetDict, load_dataset
from loguru import logger
from transformers import AutoFeatureExtractor, AutoTokenizer

from elpis.datasets.clean_text import clean_text
from elpis.models.job import Job

LOGGING_TRANSCRIPT_SAMPLE = 2


def create_dataset(job: Job) -> DatasetDict:
    if Path(job.data_args.dataset_name_or_path).is_dir():
        return create_local_dataset(job)

    return create_hf_dataset(job)


def create_local_dataset(
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


def create_hf_dataset(job: Job) -> DatasetDict:
    dataset = DatasetDict()
    data_args = job.data_args

    if job.training_args.do_train:
        dataset["train"] = load_dataset(
            data_args.dataset_name_or_path,
            data_args.dataset_config_name,
            split=data_args.train_split_name,
            token=data_args.token,
        )

        if data_args.audio_column_name not in dataset["train"].column_names:
            raise ValueError(
                f"audio_column_name '{data_args.audio_column_name}' not found"
                f" in dataset '{data_args.dataset_name_or_path}'."
                " Make sure to set `audio_column_name` to the correct audio column - one of"
                f" {', '.join(dataset['train'].column_names)}."
            )

        if data_args.text_column_name not in dataset["train"].column_names:
            raise ValueError(
                f"text_column_name {data_args.text_column_name} not found"
                f" in dataset '{data_args.dataset_name_or_path}'. "
                "Make sure to set `text_column_name` to the correct text column - one of "
                f"{', '.join(dataset['train'].column_names)}."
            )

    if job.training_args.do_eval:
        dataset["eval"] = load_dataset(
            data_args.dataset_name_or_path,
            data_args.dataset_config_name,
            split=data_args.eval_split_name,
            token=data_args.token,
        )

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
    dataset = clean_dataset(job, dataset)
    dataset = constrain_to_max_samples(job, dataset)

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

    logger.info(f"Test encoding labels: {dataset['train'][0]['labels']}")

    return dataset


def constrain_to_max_samples(job: Job, dataset: DatasetDict) -> DatasetDict:
    max_train_samples = job.data_args.max_train_samples
    max_eval_samples = job.data_args.max_eval_samples

    if job.training_args.do_train and max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    if job.training_args.do_eval and max_eval_samples is not None:
        dataset["eval"] = dataset["eval"].select(range(max_eval_samples))

    return dataset


def clean_dataset(job: Job, dataset: DatasetDict) -> DatasetDict:
    if not job.data_args.do_clean:
        return dataset

    text_column = job.data_args.text_column_name

    def clean(batch: Dict[str, Any]):
        characters_to_remove = "".join(job.data_args.chars_to_remove or [])
        characters_to_explode = "".join(job.data_args.chars_to_explode or [])

        batch[text_column] = (
            clean_text(
                batch[text_column],
                words_to_remove=job.data_args.words_to_remove,
                characters_to_remove=characters_to_remove,
                characters_to_explode=characters_to_explode,
                to_lower=job.data_args.do_lower_case or True,
            )
            + " "  # Note: not sure why this is necessary, but saw in hf docs.
        )

        return batch

    with job.training_args.main_process_first(desc="Dataset cleaning."):
        dataset = dataset.map(
            clean,
            desc="Cleaning the dataset and standardizing case.",
        )

    return dataset
