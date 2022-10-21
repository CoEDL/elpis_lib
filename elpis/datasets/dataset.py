from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from elpis.models import ElanOptions

TRANSCRIPTION_EXTENSIONS = {".eaf", ".txt"}


@dataclass
class CleaningOptions:
    """A class representing cleaning options for a dataset."""

    punctuation_to_remove: str = ""
    punctuation_to_explode: str = ""
    words_to_remove: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CleaningOptions:
        kwargs = {field.name: data[field.name] for field in fields(CleaningOptions)}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ProcessingBatch:
    """A class encapsulating the data needed for an individual processing job"""

    audio_file: Path
    transcription_file: Path
    cleaning_options: CleaningOptions
    elan_options: Optional[ElanOptions]

    def to_dict(self) -> Dict[str, Any]:
        result = {}

        result["audio_file"] = str(self.audio_file)
        result["transcription_file"] = str(self.transcription_file)
        result["cleaning_options"] = self.cleaning_options.to_dict()
        if self.elan_options is not None:
            result["elan_options"] = self.elan_options.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessingBatch:
        audio_file = Path(data["audio_file"])
        transcription_file = Path(data["transcription_file"])
        cleaning_options = CleaningOptions.from_dict(data["cleaning_options"])
        elan_options = ElanOptions.from_dict(data["elan_options"])
        return cls(
            audio_file=audio_file,
            transcription_file=transcription_file,
            cleaning_options=cleaning_options,
            elan_options=elan_options,
        )


@dataclass
class Dataset:
    """A class representing an unprocessed dataset."""

    name: str
    files: List[Path]
    cleaning_options: CleaningOptions
    elan_options: Optional[ElanOptions]

    def is_empty(self) -> bool:
        """Returns true iff the dataset contains no files."""
        return len(self.files) == 0

    def has_elan(self) -> bool:
        """Returns true iff any of the files in the dataset is an elan file."""
        return any(map((lambda file_name: file_name.suffix == ".eaf"), self.files))

    def is_valid(self) -> bool:
        """Returns true iff this dataset is valid for processing."""
        return (
            not self.is_empty()
            and len(self.files) % 2 == 0
            and len(self.mismatched_files()) == 0
            and len(self.colliding_files()) == 0
        )

    @staticmethod
    def corresponding_audio_name(transcript_file: Path) -> Path:
        """Gets the corresponding audio file name for a given transcript file."""
        return Path(transcript_file).parent / (transcript_file.stem + ".wav")

    def mismatched_files(self) -> Set[Path]:
        """Returns the list of transcript files with no corresponding
        audio and vice versa.

        Corresponding in this case means that for every transcript file with
        name x.some_extension, there is a corresponding file x.wav in the dataset.

        Returns:
            A list of the mismatched file names.
        """
        transcripts_with_audio = set(
            filter(
                lambda file: Dataset.corresponding_audio_name(file) in self.files,
                self._transcript_files(),
            )
        )
        matched_files = transcripts_with_audio | set(
            Dataset.corresponding_audio_name(file) for file in transcripts_with_audio
        )

        return set(self.files).difference(matched_files)

    def colliding_files(self) -> Set[Path]:
        """Returns the list of transcript file names that collide.

        Collide means that two transcript files would be for the same .wav
        file.

        Returns:
            A list of the colliding file names.
        """

        def would_collide(transcript_file: Path) -> bool:
            other_files = self._transcript_files().difference({transcript_file})
            other_file_names = map(lambda file: Path(file).stem, other_files)
            return Path(transcript_file).stem in other_file_names

        return set(filter(would_collide, self._transcript_files()))

    def _transcript_files(self) -> Set[Path]:
        """Returns a set of all transcription files within the dataset."""
        return set(
            filter(lambda file: file.suffix in TRANSCRIPTION_EXTENSIONS, self.files)
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Dataset:
        name = data["name"]
        files = [Path(file) for file in data["files"]]
        cleaning_options = CleaningOptions.from_dict(data["cleaning_options"])

        elan_options = None
        if "elan_options" in data:
            elan_options = ElanOptions.from_dict(data["elan_options"])

        return cls(
            name=name,
            files=files,
            cleaning_options=cleaning_options,
            elan_options=elan_options,
        )

    def to_batches(self) -> List[ProcessingBatch]:
        """Converts a valid dataset to a list of processing jobs, matching
        transcript and audio files.
        """
        return [
            ProcessingBatch(
                transcription_file=transcription_file,
                audio_file=self.corresponding_audio_name(transcription_file),
                cleaning_options=self.cleaning_options,
                elan_options=self.elan_options,
            )
            for transcription_file in self._transcript_files()
        ]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "files": [file.name for file in self.files],
            "cleaning_options": self.cleaning_options.to_dict(),
        }

        if self.elan_options is not None:
            result["elan_options"] = self.elan_options.to_dict()

        return result
