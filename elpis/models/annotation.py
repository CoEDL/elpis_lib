from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Annotation:
    """A class which represents a section of speech for a given audio file and
    sample rate. If start_ms and end_ms aren't specified, it is assumed that the
    Annotation spans the entire audio file.
    """

    audio_file: Path
    transcript: str
    start_ms: Optional[int] = None
    stop_ms: Optional[int] = None

    def is_timed(self) -> bool:
        """Returns true iff the annotation exists between a start and stop time for
        the given recording.
        """
        return self.start_ms is not None and self.stop_ms is not None

    def to_dict(self) -> Dict[str, Any]:
        """Converts an annotation to a serializable dictionary"""
        result = dict(self.__dict__)
        result["audio_file"] = str(self.audio_file)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Annotation:
        """Builds an annotation from a serializable dictionary

        Throws an error if the required keys are not found.
        """
        return cls(
            audio_file=Path(data["audio_file"]),
            transcript=data["transcript"],
            start_ms=data.get("start_ms"),
            stop_ms=data.get("stop_ms"),
        )
