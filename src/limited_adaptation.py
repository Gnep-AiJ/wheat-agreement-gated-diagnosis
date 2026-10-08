"""Small shared helpers for the fixed matched adaptation experiment."""
import hashlib

import numpy as np
import torch


def affine_head(scaler, classifiers):
    coefficients = np.concatenate([m.coef_ for m in classifiers], axis=0)
    intercepts = np.concatenate([m.intercept_ for m in classifiers])
    weight = coefficients / scaler.scale_[None, :]
    bias = intercepts - weight @ scaler.mean_
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight.astype(np.float32)))
        layer.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    return layer


def parameter_hash(model, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def accumulation_denominator(start, n, effective_batch=16, labels=6):
    group_start = (start // effective_batch) * effective_batch
    return min(effective_batch, n - group_start) * labels


def classification_metrics(probabilities, truth):
    p, y = np.asarray(probabilities), np.asarray(truth, dtype=bool)
    assert p.shape == y.shape and p.ndim == 2 and p.shape[1] == 6
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    top = p.argmax(1)
    correct = y[np.arange(len(y)), top]
    recall = [float((top[y[:, j]] == j).mean()) if y[:, j].any() else None for j in range(6)]
    predicted_set = p >= .5
    predicted_healthy = predicted_set[:, 0] & ~predicted_set[:, 1:].any(1)
    diseased = y[:, 1:].any(1)
    return {"n": len(y), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "named_top1_recall": recall, "macro_named_top1_recall": float(np.mean([v for v in recall if v is not None])),
            "exact_set_accuracy_at_0_5": float((predicted_set == y).all(1).mean()),
            "diseased_n": int(diseased.sum()), "healthy_only_false_reassurance_at_0_5": int((predicted_healthy & diseased).sum()),
            "empty_sets_at_0_5": int((~predicted_set.any(1)).sum()),
            "healthy_and_disease_conflicts_at_0_5": int((predicted_set[:, 0] & predicted_set[:, 1:].any(1)).sum())}
