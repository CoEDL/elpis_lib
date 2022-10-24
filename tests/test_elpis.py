import os
from pathlib import Path

from loguru import logger
from pytest import mark

from elpis import __version__
from elpis.datasets import Dataset
from elpis.datasets.dataset import CleaningOptions
from elpis.datasets.preprocessing import process_batch
from elpis.models.elan_options import ElanOptions, ElanTierSelector
from elpis.trainer.job import TrainingJob, TrainingOptions
from elpis.trainer.trainer import train
from elpis.transcriber.results import build_elan, build_text
from elpis.transcriber.transcribe import build_pipeline, transcribe

DATA_DIR = Path(__file__).parent / "data"
ABUI_DIR = DATA_DIR / "abui"
FILES = [ABUI_DIR / name for name in os.listdir(ABUI_DIR)]

DATASET = Dataset(
    name="dataset",
    files=FILES,
    cleaning_options=CleaningOptions(),
    elan_options=ElanOptions(
        selection_mechanism=ElanTierSelector.NAME, selection_value="Phrase"
    ),
)


def test_version():
    assert __version__ == "0.1.0"


@mark.integration
def test_everything(tmp_path: Path):
    logger.info(f"folder: {tmp_path}")
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"

    # Make all directories
    for directory in dataset_dir, model_dir, output_dir:
        directory.mkdir()

    # Preprocessing
    batches = DATASET.to_batches()
    for batch in batches:
        process_batch(batch, dataset_dir)

    # Train the model
    job = TrainingJob(
        model_name="model",
        dataset_name="dataset",
        options=TrainingOptions(epochs=2, learning_rate=0.001),
    )
    train(
        job=job,
        output_dir=model_dir,
        dataset_dir=dataset_dir,
    )

    # Perform inference with pipeline
    asr = build_pipeline(
        pretrained_location=str(model_dir.absolute()),
    )
    audio = DATA_DIR / "oily_rag.wav"
    annotations = transcribe(audio, asr)

    # Build output files
    text_file = output_dir / "test.txt"
    with open(text_file, "w") as output_file:
        output_file.write(build_text(annotations))

    elan_file = output_dir / "test.eaf"
    eaf = build_elan(annotations)
    eaf.to_file(str(elan_file))

    logger.success("Finished everything ;)")
