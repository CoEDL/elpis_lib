from pathlib import Path

from transformers import TrainingArguments

from elpis.models.job import DataArguments, Job, ModelArguments
from elpis.trainer.trainer import run_job
from elpis.transcriber.results import build_elan, build_text
from elpis.transcriber.transcribe import build_pipeline, transcribe

DATASETS_PATH = Path(__file__).parent.parent.parent / "datasets"
DIGITS_PATH = DATASETS_PATH / "digits-preview"
TRANSCRIBE_AUDIO = DIGITS_PATH / "test/audio2.wav"

print("------ Training files ------")
# Setup
tmp_path = Path("testdir")
# tmp_path = Path("/tmp") / "testscript"
model_dir = tmp_path / "model"
output_dir = tmp_path / "output"

# Make all directories
for directory in model_dir, output_dir:
    directory.mkdir(exist_ok=True, parents=True)

# Train the model
job = Job(
    model_args=ModelArguments(
        "facebook/wav2vec2-base",
        attention_dropout=0.1,
        layerdrop=0.0,
        freeze_feature_encoder=True,
    ),
    data_args=DataArguments(
        dataset_name_or_path="mozilla-foundation/common_voice_11_0",
        dataset_config_name="gn",
        train_split_name="train",
        text_column_name="sentence",
        audio_column_name="audio",
        min_duration_in_seconds=1,
        max_duration_in_seconds=20,
        max_train_samples=100,
        do_clean=True,
        # stream_dataset=True,
    ),
    training_args=TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        num_train_epochs=20,
        learning_rate=1e-4,
        group_by_length=True,
        logging_steps=10,
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
