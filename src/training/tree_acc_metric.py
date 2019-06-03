import torch
import numpy as np

from typing import Optional, Callable, List
from allennlp.training.metrics import Metric
from utils.logical_eval.tree import is_tree_eq

class TreeAccuracy(Metric):
    def __init__(self, decode_fn: Callable[[torch.Tensor], List[List[str]]]):
        self._correct = 0
        self._total = 0
        self._decode_fn = decode_fn

    def reset(self):
        self._correct = 0
        self._total = 0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional):
        predictions, gold_labels = map(lambda x: x.detach().cpu().numpy(), [predictions, gold_labels])
        pred_tokens, gold_tokens = map(self._decode_fn, (predictions, gold_labels))

        for x, y in zip(pred_tokens, gold_tokens):
            self._total += 1

            if is_tree_eq(" ".join(x), " ".join(y), not_layout=True):
                self._correct += 1

    def get_metric(self, reset: bool):
        acc = float(self._correct) / self._total
        if reset:
            self.reset()

        return {"TreeAcc": acc}




