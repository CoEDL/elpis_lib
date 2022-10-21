import re
from typing import List, Optional


def clean_text(
    text: str,
    words_to_remove: Optional[List[str]] = None,
    punctuation_to_explode: str = "",
    punctuation_to_remove: str = "",
) -> str:
    """Cleans the text based on the supplied options.

    Parameters:
        text: The text to clean.
        options: The cleaning options.

    Returns:
        The cleaned text
    """
    words = text.lower().split()

    if words_to_remove is not None:
        words = filter(lambda word: word not in words_to_remove, words)

    if punctuation_to_explode != "":
        words = map(lambda word: explode(word, punctuation_to_explode), words)

    if punctuation_to_remove != "":
        words = map(lambda word: collapse(word, punctuation_to_remove), words)

    result = " ".join(words).strip()
    return remove_consecutive_spaces(result)


def explode(text: str, pattern: str) -> str:
    """Replace occurences of the pattern with spaces within the given text.

    Parameters:
        text: The text to modify.
        pattern: The pattern of characters to replace with spaces.

    Returns:
        The text with instances of the pattern exploded.
    """
    pattern = re.escape(pattern)
    return re.sub(rf"[{pattern}]", " ", text)


def collapse(text: str, pattern: str) -> str:
    """Remove occurences of the pattern within the given text.

    Parameters:
        text: The text to modify.
        pattern: The pattern of characters to remove.

    Returns:
        The text with instances of the pattern removed.
    """
    pattern = re.escape(pattern)
    return re.sub(rf"[{pattern}]", "", text)


def remove_consecutive_spaces(text: str) -> str:
    """Replace consecutive spaces with a single one in some given text.

    Parameters:
        text: The text to modify.

    Returns
        The supplied text with conseucutive spaces reduced to one.
    """
    return re.sub("[ ]+", " ", text)
