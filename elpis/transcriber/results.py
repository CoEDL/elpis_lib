from typing import List

from pympi.Elan import Eaf

from elpis.models.annotation import Annotation


def build_elan(annotations: List[Annotation], tier_name: str = "Phrase") -> Eaf:
    """Builds an elan file from the given annotations and tier_name.

    Parameters:
        annotations: The list of annotations to add.
        tier_name: The name of the tier to add the annotations under.

    Returns:
        The resulting elan file.
    """
    result = Eaf()
    result.add_tier(tier_name)
    for annotation in annotations:
        result.add_annotation(
            id_tier=tier_name,
            start=annotation.start_ms,
            end=annotation.stop_ms,
            value=annotation.transcript,
        )
    return result


def build_text(annotations: List[Annotation]) -> str:
    """Combines all the text from a list of annotations, ordered from earliest
    start time.

    Parameters:
        annotations: The list of annotations to Combines

    Returns:
        The combined transcripts.
    """
    annotations = sorted(
        annotations,
        key=(
            lambda annotation: annotation.start_ms
            if annotation.start_ms is not None
            else 0
        ),
    )
    return " ".join(annotation.transcript for annotation in annotations)
