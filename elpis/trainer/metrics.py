from typing import Callable, Dict, Optional, Sequence

import evaluate
from transformers import EvalPrediction
from transformers.integrations import np


def create_metrics(
    metric_names: Sequence[str],
) -> Optional[Callable[[EvalPrediction], Dict]]:
    # Handle metrics
    if len(metric_names) == 0:
        return

    metrics = evaluate.combine(evaluate.load(metric) for metric in metric_names)

    def compute_metrics(pred: EvalPrediction) -> Dict:
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        return metrics.compute(predictions=predictions, references=labels)

    return compute_metrics
