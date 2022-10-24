# Elpis Core Library

The Core Elpis Library, providing a quick api to [:hugs: transformers](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads)
for automatic-speech-recognition.

You can use the library to:

- Perform standalone inference using a pretrained HFT model.
- Fine tune a pretrained ASR model on your own dataset.
- Generate text and Elan files from inference results for further analysis.

## Documentation

Documentation for the library can be be found [here](https://coedl.github.io/elpis_lib/index.html).

## Dependencies

While we try to be as machine-independant as possible, there are some dependencies
you should be aware of when using this library:

- Processing datasets (`elpis.datasets.processing`) requires `librosa`, which
  depends on having `libsndfile` installed on your computer. If you're using
  elpis within a docker container, you may have to manually install
  `libsndfile`.
- Transcription (`elpis.transcription.transcribe`) requires `ffmpeg` if your
  audio you're attempting to transcribe needs to be resampled before it can
  be used. The default sample rate we assume is 16khz.
- The preprocessing flow (`elpis.datasets.preprocessing`) is free of external
  dependencies.

## Installation

You can install the elpis library with:
`pip3 install elpis`

## Usage

Below are some typical examples of use cases

### Standalone Inference

```python
from pathlib import Path

from elpis.transcriber.results import build_text
from elpis.transcriber.transcribe import build_pipeline, transcribe

# Perform inference
asr = build_pipeline(pretrained_location="facebook/wav2vec2-base-960h")
audio = Path("<to_some_audio_file.wav>")
annotations = transcribe(audio, asr) # Timed, per word annotation data

result = build_text(annotations) # Combine annotations to extract all text
print(result)

# Build output files
text_file = output_dir / "test.txt"
with open(text_file, "w") as output_file:
    output_file.write(result)
```

### Fine-tuning a Pretrained Model on Local Dataset

```python
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

files: List[Path] = [...] # A list of paths to the files to include.

dataset = Dataset(
    name="dataset",
    files=files,
    cleaning_options=CleaningOptions(), # Default cleaning options
    # Elan data extraction info- required if dataset includes .eaf files.
    elan_options=ElanOptions(
        selection_mechanism=ElanTierSelector.NAME, selection_value="Phrase"
    ),
)

# Setup
tmp_path = Path('...')

dataset_dir = tmp_path / "dataset"
model_dir = tmp_path / "model"
output_dir = tmp_path / "output"

# Make all directories
for directory in dataset_dir, model_dir, output_dir:
    directory.mkdir(exist_ok=True, parents=True)

# Preprocessing
batches = dataset.to_batches()
for batch in batches:
    process_batch(batch, dataset_dir)

# Train the model
job = TrainingJob(
    model_name="some_model",
    dataset_name="some_dataset",
    options=TrainingOptions(epochs=2, learning_rate=0.001),
    base_model="facebook/wav2vec2-base-960h"
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
audio = Path("<to_some_audio_file.wav>")
annotations = transcribe(audio, asr)

# Build output files
text_file = output_dir / "test.txt"
with open(text_file, "w") as output_file:
    output_file.write(build_text(annotations))

elan_file = output_dir / "test.eaf"
eaf = build_elan(annotations)
eaf.to_file(str(elan_file))

print('voila ;)')
```
