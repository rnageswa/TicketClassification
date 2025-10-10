# src/utils/metrics.py
import torchmetrics
import numpy as np
import torch

def compute_triage_metrics(eval_pred, num_labels):
    """
    Computes classification metrics for triage model evaluation.
    This is a helper function, the train script uses torchmetrics directly.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = torchmetrics.functional.f1_score(
        torch.from_numpy(predictions),
        torch.from_numpy(labels),
        task='multiclass',
        num_classes=num_labels,
        average='weighted'
    )
    accuracy = torchmetrics.functional.accuracy(
        torch.from_numpy(predictions),
        torch.from_numpy(labels),
        task='multiclass',
        num_classes=num_labels
    )
    
    return {"f1": f1.item(), "accuracy": accuracy.item()}
