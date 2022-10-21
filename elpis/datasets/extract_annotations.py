from itertools import chain
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from pympi.Elan import Eaf

from elpis.models import Annotation, ElanOptions, ElanTierSelector


def extract_annotations(
    transcription_file: Path, elan_options: Optional[ElanOptions] = None
) -> List[Annotation]:
    """Extracts annotations from the supplied transcription file.

    If the transcription file is an elan file, elan_options is required.

    Parameters:
        transcription_file: The file from which to extract annotations
        elan_options: Options to include for determining how to extract annotations
                from elan data.

    Returns:
        A list of found annotations.
        Returns an empty list if there was a problem.
    """
    if transcription_file.suffix == ".txt":
        return extract_text_annotations(transcription_file)

    if transcription_file.suffix != ".eaf":
        logger.error(f"Unrecognised file format: {transcription_file}")
        return []

    if elan_options is None:
        logger.error(f"Missing elan options for extraction job.")
        return []

    return extract_elan_annotations(
        transcription_file,
        selection_type=elan_options.selection_mechanism,
        selection_data=elan_options.selection_value,
    )


def extract_text_annotations(file: Path) -> List[Annotation]:
    """Extract transcription information from a text file.

    Parameters:
        file_name: The name of the downloaded file.

    Returns:
        A list of utterance information for the given file.
    """
    with open(file) as transcription_file:
        transcription = transcription_file.read()

    return [
        Annotation(
            audio_file=file.parent / (file.stem + ".wav"),
            transcript=transcription,
        )
    ]


def extract_elan_annotations(
    elan_file_path: Path, selection_type: ElanTierSelector, selection_data: str
) -> List[Annotation]:
    """Extracts annotations from a particular tier in an eaf file (ELAN
    Annotation Format).

    Tiers are nodes from the tree structure in the .eaf file.
    The tier to read from is determined by tier order (eg top tier would be order 1),
    tier type (eg default-lt) or tier name (eg Phrase).

    Parameters:
        elan_file_path: The path to the eaf file.
        selection_type: The method of determining which tier data to extract.
        selection_data: The data corresponding to the selection_type.

    Returns:
        A list of the annotations contained for the supplied data. Returns an
        empty list if the given selection isn't found.
    """
    logger.info(
        f"processing eaf {elan_file_path} using {selection_type}: {selection_data}"
    )

    match selection_type:
        case ElanTierSelector.NAME:
            return get_annotations_by_tier_name(elan_file_path, selection_data)
        case ElanTierSelector.TYPE:
            return get_annotations_by_tier_type(elan_file_path, selection_data)
        case ElanTierSelector.ORDER:
            try:
                order = int(selection_data)
            except:
                order = 1
            return get_annotations_by_tier_order(elan_file_path, order)


def get_annotations_by_tier_order(
    elan_file_path: Path, tier_order: int
) -> List[Annotation]:
    """Retrieves all annotations for a given tier order within an eaf file.

    Parameters:
        elan_file_path: The path to the eaf file.
        tier_order: The tier order to extract from (starts at 1)

    Returns:
        A list of the annotations contained for the supplied tier order.
        Returns an empty list if the given tier order exceeds the nesting of
        the file.
    """
    elan = Eaf(elan_file_path)

    tier_names: List[str] = list(elan.get_tier_names())
    if tier_order > len(tier_names):
        logger.error(
            f"tier_order: {tier_order} exceeds tier length for {elan_file_path}"
        )
        return []

    tier_name = tier_names[tier_order - 1]
    return get_annotations_by_tier_name(
        elan_file_path=elan_file_path, tier_name=tier_name
    )


def get_annotations_by_tier_type(
    elan_file_path: Path, tier_type: str
) -> List[Annotation]:
    """Retrieves all annotations for a given linguistic tier type in an eaf file.

    Parameters:
        elan_file_path: The path to the eaf file.
        tier_type: The linguistic type from which to extract Annotation data.

    Returns:
        A list of the annotations contained for the supplied linguistic type.
        Returns an empty list if the type is not found.
    """
    elan = Eaf(elan_file_path)

    if tier_type not in list(elan.get_linguistic_type_names()):
        logger.error(f"tier_type: {tier_type} not found in file: {elan_file_path}")
        return []

    tier_names = elan.get_tier_ids_for_linguistic_type(tier_type)
    annotations = (
        get_annotations_by_tier_name(elan_file_path, name) for name in tier_names
    )
    # Flatten list of annotations
    return list(chain(*annotations))


def get_annotations_by_tier_name(
    elan_file_path: Path, tier_name: str
) -> List[Annotation]:
    """Retrieves all annotations for a given tier name in an eaf file.

    Parameters:
        elan_file_path: The path to the eaf file.
        tier_name: The tier name from which to extract Annotation data.

    Returns:
        A list of the annotations contained for the supplied tier name.
        Returns an empty list if the name is not found.
    """
    elan = Eaf(elan_file_path)

    if tier_name not in list(elan.get_tier_names()):
        logger.error(f"tier_name: {tier_name} not found in file {elan_file_path}")
        return []

    def create_annotation(elan_annotation: Tuple[str, str, str]) -> Annotation:
        start, end, transcript = elan_annotation
        return Annotation(
            audio_file=elan_file_path.parent / (elan_file_path.stem + ".wav"),
            transcript=transcript,
            start_ms=int(start),
            stop_ms=int(end),
        )

    return list(map(create_annotation, elan.get_annotation_data_for_tier(tier_name)))
