import json
from pathlib import Path

import pytest

from elpis.models import VOCAB_FILE, Vocab


@pytest.fixture()
def base() -> Vocab:
    return Vocab({"a": 0, "b": 1})


@pytest.fixture()
def other() -> Vocab:
    return Vocab({"c": 0, "b": 1, "d": 2})


def test_build_vocab_from_set():
    symbols = {"a", "b", "<unk>", "e"}
    assert Vocab.from_set(symbols).symbols == symbols


def test_build_vocab_from_strings():
    strings = ("hello", "there")
    expected = set("".join(strings))
    assert Vocab.from_strings(strings).symbols == expected


def test_replace_vocab(base: Vocab):
    assert base.vocab["a"] == 0
    base.replace("a", "c")

    assert "a" not in base.vocab
    assert base.vocab["c"] == 0


def test_add_vocab(base: Vocab):
    base.add("<unk>")
    assert base.vocab["<unk>"] == 2


def test_merge_vocab(base: Vocab, other: Vocab):
    merged = base.merge(other)
    assert merged.vocab == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_vocab_symbols(base: Vocab):
    assert base.symbols == set(("a", "b"))


def test_save_vocab(base: Vocab, tmp_path: Path):
    file_path = tmp_path / "nice.json"

    def check_valid_file(path: Path, vocab: Vocab):
        with open(path, "r") as check:
            assert json.load(check) == vocab.vocab

    base.save(tmp_path)
    expected_file = tmp_path / VOCAB_FILE
    check_valid_file(expected_file, base)

    base.save(file_path)
    check_valid_file(file_path, base)
