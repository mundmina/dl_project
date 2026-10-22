"""Predictions under every test condition and the metrics built from them.

A predictions table has one row per image: image_id, label, species, pred_<condition>...
All metrics are computed from that table, so ResNet and HOG + SVM are scored identically.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .datasets import CONDITIONS
from .project import make_dataset
from .train import make_loader


def predictions_table(data, split, predict_fn, conditions=CONDITIONS):
    """predict_fn(condition) -> array of predicted labels aligned with data.splits[split]."""
    df = data.splits[split][["image_id", "label", "breed", "species"]].copy()
    for condition in conditions:
        df[f"pred_{condition}"] = predict_fn(condition)
    return df


def torch_predict_fn(model, data, split, device, batch_size=128, num_workers=2, **dataset_kwargs):
    @torch.no_grad()
    def predict(condition):
        loader = make_loader(make_dataset(data, split, condition=condition, **dataset_kwargs),
                             batch_size, False, 0, num_workers)
        model.eval()
        preds = []
        for x, _ in loader:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        return torch.cat(preds).numpy()
    return predict


def summarize(table):
    """Accuracy / macro F1 per condition, robustness gaps and flip rates."""
    out = {}
    conditions = [c[len("pred_"):] for c in table.columns if c.startswith("pred_")]
    for c in conditions:
        out[f"acc_{c}"] = accuracy_score(table["label"], table[f"pred_{c}"])
        out[f"f1_{c}"] = f1_score(table["label"], table[f"pred_{c}"], average="macro")
    for c, short in (("swapped", "swap"), ("fgonly", "fg")):
        if c in conditions:
            out[f"gap_{short}"] = out["acc_original"] - out[f"acc_{c}"]
            # share of images whose prediction changes when only the background changes
            out[f"flip_{short}"] = float((table["pred_original"] != table[f"pred_{c}"]).mean())
    return out


def aggregate(summaries):
    """{run_name: summary} for several seeds of one model -> mean and std per metric."""
    df = pd.DataFrame(summaries).T
    return pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=1) if len(df) > 1 else np.nan, "n": len(df)})


def per_class_accuracy(table, condition):
    correct = table["label"] == table[f"pred_{condition}"]
    return correct.groupby(table["breed"]).mean().sort_values()
