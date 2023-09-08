from typing import Callable, Dict, Optional, Sequence

import evaluate
import numpy as np
from loguru import logger
from transformers import EvalPrediction, Wav2Vec2Processor


def create_metrics(
    metric_names: Sequence[str], processor: Wav2Vec2Processor
) -> Optional[Callable[[EvalPrediction], Dict]]:
    # Handle metrics
    if len(metric_names) == 0:
        return

    # Note: was using evaluate.combine but was having many unexpected errors.
    metrics = {name: evaluate.load(name) for name in metric_names}

    def compute_metrics(pred: EvalPrediction) -> Dict:
        # taken from https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
        pred_logits = pred.predictions

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id  # type: ignore

        # Taken from: https://discuss.huggingface.co/t/code-review-compute-metrics-for-wer-with-wav2vec2processorwithlm/16841/3
        if type(processor).__name__ == "Wav2Vec2ProcessorWithLM":
            pred_str = processor.batch_decode(pred_logits).text
        else:
            pred_ids = np.argmax(pred_logits, axis=-1)
            pred_str = processor.batch_decode(pred_ids)

        # We do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        logger.debug(f"METRICS->pred: {pred_str} label:{label_str}")

        result = {name: metric.compute(predictions=pred_str, references=label_str) for name, metric in metrics.items()} 
        logger.debug(f"Metrics Result: {result}")
        return result 

    return compute_metrics
