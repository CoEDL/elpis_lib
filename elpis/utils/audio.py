from pathlib import Path

from pedalboard.io import ReadableAudioFile, WriteableAudioFile


def get_sample_rate(audio_path: Path) -> int:
    """Gets the current sample rate of the given audio file.

    Parameters:
        audio_path: The path to the audio file.

    Returns:
        The sample rate of the given file.
    """
    with ReadableAudioFile(str(audio_path)) as audio_file:
        return int(audio_file.samplerate)


def resample(audio_path: Path, destination: Path, sample_rate: int) -> None:
    """Copies a wav file to the destination, with the given
    sample rate.

    Parameters:
        audio_path (Path): The path of the file to resample
        destination (Path): The destination at which to create the resampled file
        sample_rate (int): The sample rate for the resampled audio.
    """
    with ReadableAudioFile(str(audio_path)).resampled_to(sample_rate) as audio_file:
        data = audio_file.read(audio_file.frames)
        num_channels = audio_file.num_channels

    with WriteableAudioFile(
        str(destination), samplerate=sample_rate, num_channels=num_channels
    ) as destination_file:
        destination_file.write(data)


def cut(audio_path: Path, destination: Path, start_ms: int, stop_ms: int) -> None:
    """Creates a new wav file at the destination, restricted to the given start
    and stop times.

    Parameters:
        audio_path (Path): The path of the file to resample.
        destination (Path): The destination at which to create the resampled file.
        start_ms (int): The start time in milliseconds to record from.
        stop_ms (int):  The stop time in milliseconds to record to.
    """

    with ReadableAudioFile(str(audio_path)) as audio_file:
        start = int(start_ms * audio_file.samplerate / 1000)
        stop = int(stop_ms * audio_file.samplerate / 1000)

        audio_file.seek(start)
        data = audio_file.read(stop - start)
        num_channels = audio_file.num_channels
        sample_rate = audio_file.samplerate

    with WriteableAudioFile(
        str(destination), samplerate=sample_rate, num_channels=num_channels
    ) as destination_file:
        destination_file.write(data)
