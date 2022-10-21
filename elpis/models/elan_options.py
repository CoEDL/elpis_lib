from dataclasses import dataclass
from enum import Enum
from typing import Dict


class ElanTierSelector(Enum):
    """A class representing a method of selecting elan tiers"""

    ORDER = "tier_order"
    TYPE = "tier_type"
    NAME = "tier_name"


@dataclass
class ElanOptions:
    """A class representing options for how to extract utterance information
    from an elan file."""

    selection_mechanism: ElanTierSelector
    selection_value: str

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ElanOptions":
        return cls(
            selection_mechanism=ElanTierSelector(data["selection_mechanism"]),
            selection_value=data["selection_value"],
        )

    def to_dict(self) -> Dict[str, str]:
        result = dict(self.__dict__)
        result["selection_mechanism"] = self.selection_mechanism.value
        return result
