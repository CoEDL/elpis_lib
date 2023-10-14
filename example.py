import shutil
from itertools import groupby, takewhile
from pathlib import Path
from pprint import pprint

from loguru import logger
from tqdm import tqdm
from transformers import TrainingArguments, training_args

from elpis.datasets import Dataset
from elpis.datasets.dataset import CleaningOptions
from elpis.datasets.preprocessing import process_batch
from elpis.models import ElanOptions, ElanTierSelector
from elpis.models.job import DataArguments, Job, ModelArguments
from elpis.trainer.trainer import run_job
from elpis.transcriber.results import build_elan, build_text
from elpis.transcriber.transcribe import build_pipeline, transcribe

DATASETS_PATH = Path(__file__).parent.parent.parent / "datasets"
TIMIT_PATH = DATASETS_PATH / "timit"
DIGITS_PATH = DATASETS_PATH / "digits-preview"

TRAINING_FILES = list((DIGITS_PATH / "train").rglob("*.*"))
TRANSCRIBE_AUDIO = DIGITS_PATH / "test/audio2.wav"

TIMIT_TIER_NAME = "default"
TIER_NAME = "tx"

print("------ Training files ------")
# print(training_files)
DIGITS_DATASET = Dataset(
    name="my_dataset",
    files=TRAINING_FILES,
    cleaning_options=CleaningOptions(),  # Default cleaning options
    # Elan data extraction info - required if dataset includes .eaf files.
    elan_options=ElanOptions(
        selection_mechanism=ElanTierSelector.NAME, selection_value=TIER_NAME
    ),
)

TIMIT_DATASET = Dataset(
    name="my_dataset",
    files=list(TIMIT_PATH.rglob("*.*")),
    cleaning_options=CleaningOptions(),  # Default cleaning options
    # Elan data extraction info - required if dataset includes .eaf files.
    elan_options=ElanOptions(
        selection_mechanism=ElanTierSelector.NAME, selection_value="default"
    ),
)

dataset = TIMIT_DATASET

# Setup
tmp_path = Path("testdir")
# tmp_path = Path("/tmp") / "testscript"
dataset_dir = tmp_path / "dataset"
model_dir = tmp_path / "model"
output_dir = tmp_path / "output"
cache_dir = tmp_path / "cache"

# Reset dir between runs
if tmp_path.exists():
    shutil.rmtree(tmp_path)

# Make all directories
for directory in dataset_dir, model_dir, output_dir:
    directory.mkdir(exist_ok=True, parents=True)

# Preprocessing
logger.info("Creating batches")
batches = dataset.to_batches()
logger.info("Processing batches")
for batch in tqdm(batches, unit="batch"):
    process_batch(batch, dataset_dir)

# Train the model
job = Job(
    model_args=ModelArguments(
        "facebook/wav2vec2-base",
        # ctc_zero_infinity=True,
        attention_dropout=0.1,
        # hidden_dropout=0.1,
        # mask_time_prob=0.05,
        layerdrop=0.0,
        freeze_feature_encoder=True,
    ),
    data_args=DataArguments(
        dataset_name_or_path=str(dataset_dir),
        text_column_name="transcript",
        min_duration_in_seconds=1,
        max_duration_in_seconds=5,
    ),
    training_args=TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        # evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        num_train_epochs=20,
        learning_rate=1e-4,
        group_by_length=True,
        weight_decay=0.005,
        warmup_steps=1000,
        logging_steps=10,
        eval_steps=100,
        save_steps=400,
        save_total_limit=2,
    ),
)


print("------ JOB ------")
print(job)
run_job(job)
print("------ TRAINED ------")

# Perform inference with pipeline
print("------ INFER ------")
asr = build_pipeline(
    pretrained_location=str(model_dir.absolute()),
)

annotations = transcribe(TRANSCRIBE_AUDIO, asr)
print(build_text(annotations))

# Build output files
print("------ OUTPUT ------")
text_file = output_dir / "test.txt"

with open(text_file, "w") as output_file:
    output_file.write(build_text(annotations))

elan_file = output_dir / "test.eaf"
eaf = build_elan(annotations)
eaf.to_file(str(elan_file))

print("voila ;)")
