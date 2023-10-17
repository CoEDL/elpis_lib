import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from datasets import DatasetDict

VOCAB_FILE = "vocab.json"


@dataclass
class Vocab:
    """A class which represents a dictionary of encountered tokens in a dataset."""

    vocab: Dict[str, int]

    @property
    def symbols(self) -> Set[str]:
        return set(self.vocab.keys())

    def merge(self, other: "Vocab") -> "Vocab":
        """Creates a new Vocab which includes all symbols in the merged two."""
        vocab = self.symbols | other.symbols
        return Vocab.from_set(vocab)

    def save(self, path: Path) -> None:
        """Saves the vocab to the supplied path.

        If the path is a folder, saves as vocab.json, within it.
        """
        if path.is_dir():
            path /= VOCAB_FILE

        with open(path, "w") as out:
            json.dump(self.vocab, out)

    def add(self, char: str) -> None:
        """Adds a new character into the vocab."""
        if char in self.vocab:
            return

        self.vocab[char] = len(self.vocab)

    def replace(self, original: str, replacement: str) -> None:
        """Replaces the supplied character mapping in the vocab."""
        if original not in self.vocab or original == replacement:
            return

        self.vocab[replacement] = self.vocab[original]
        self.vocab.pop(original)

    @classmethod
    def from_set(cls, symbols: Set[str]) -> "Vocab":
        """Builds a vocab from a set of symbols."""
        vocab = {symbol: index for index, symbol in enumerate(sorted(symbols))}
        return cls(vocab=vocab)

    @classmethod
    def from_strings(cls, texts: Iterable[str]) -> "Vocab":
        """Builds an vocab from a iterable text collection."""

        def reducer(result: Set[str], text: str) -> Set[str]:
            return result | set(text)

        symbols = reduce(reducer, texts, set())
        return cls.from_set(symbols)
