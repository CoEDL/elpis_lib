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
