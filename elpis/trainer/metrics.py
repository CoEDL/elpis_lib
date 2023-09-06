from typing import Callable, Dict, Optional, Sequence

import evaluate
import numpy as np
from tokenizers import Tokenizer
from transformers import EvalPrediction, Wav2Vec2Processor


def create_metrics(
    metric_names: Sequence[str], processor: Wav2Vec2Processor
) -> Optional[Callable[[EvalPrediction], Dict]]:
    # Handle metrics
    if len(metric_names) == 0:
        return

    metrics = evaluate.combine([evaluate.load(metric) for metric in metric_names])

    def compute_metrics(pred: EvalPrediction) -> Dict:
        # taken from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id  # type: ignore

        pred_str = processor.batch_decode(pred_ids)

        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        return metrics.compute(predictions=pred_str, references=label_str)

    return compute_metrics
