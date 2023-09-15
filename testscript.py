import shutil
from pathlib import Path
from typing import List

from elpis.datasets import Dataset
from elpis.datasets.dataset import CleaningOptions
from elpis.datasets.preprocessing import process_batch
from elpis.models import ElanOptions, ElanTierSelector
from elpis.trainer.job import TrainingJob, TrainingOptions
from elpis.trainer.trainer import train
from elpis.transcriber.results import build_elan, build_text
from elpis.transcriber.transcribe import build_pipeline, transcribe

TIMIT_PATH = Path("../../datasets/timit")
DIGITS_PATH = Path("../../datasets/digits-preview")

TRAINING_FILES = list((DIGITS_PATH / "train").rglob("*.*"))
TRANSCRIBE_AUDIO = DIGITS_PATH / "test/audio2.wav"

TIMIT_TIER_NAME = "default"
TIER_NAME = "tx"

print("------ Training files ------")
# print(training_files)
dataset = Dataset(
    name="my_dataset",
    files=TRAINING_FILES,
    cleaning_options=CleaningOptions(),  # Default cleaning options
    # Elan data extraction info - required if dataset includes .eaf files.
    elan_options=ElanOptions(
        selection_mechanism=ElanTierSelector.NAME, selection_value=TIER_NAME
    ),
)

# Setup
tmp_path = Path("/tmp") / "testscript"
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
batches = dataset.to_batches()
for batch in batches:
    process_batch(batch, dataset_dir)

# Train the model
job = TrainingJob(
    model_name="my_model",
    dataset_name="my_dataset",
    options=TrainingOptions(
        batch_size=4, epochs=20, learning_rate=3e-7, freeze_feature_extractor=True
    ),
    base_model="facebook/wav2vec2-base-960h",
)

print("------ JOB ------")
print(job)
train(job=job, output_dir=model_dir, dataset_dir=dataset_dir, cache_dir=cache_dir)
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
