"""Analysis on saved prediction tables: bootstrap confidence intervals, per-breed gaps, flip examples."""
import numpy as np
import pandas as pd

# Background-swap probability of every trained configuration (ablation axis).
P_SWAP = {"resnet_A": 0.0, "resnet_p025": 0.25, "resnet_B": 0.5, "resnet_p100": 1.0}

# Architecture comparison: label -> (model without swap, model with swap p = 0.5).
FAMILIES = {
    "ResNet-18": ("resnet_A", "resnet_B"),
    "ResNet-18 (scratch)": ("scratch_A", "scratch_B"),
    "ResNet-50": ("resnet50_A", "resnet50_B"),
    "ConvNeXt-T": ("convnext_A", "convnext_B"),
    "Swin-T": ("swin_A", "swin_B"),
}

# Ways to handle the background during training, all on pretrained ResNet-18.
METHODS = {
    "None (A)": "resnet_A",
    "CutMix": "resnet_cutmix",
    "Blurred background": "resnet_blur",
    "Gray background": "resnet_gray",
    "Saliency penalty, λ = 1": "resnet_rrr1",
    "Saliency penalty, λ = 10": "resnet_rrr10",
    "Places365 swap (B)": "resnet_B",
}


def model_name(run_name):
    return run_name.rsplit("_seed", 1)[0] if "_seed" in run_name else run_name


def correctness(tables, model, condition):
    """images x seeds matrix: 1 if that seed's prediction is correct."""
    cols = [(t[f"pred_{condition}"] == t["label"]).to_numpy(float)
            for run, t in sorted(tables.items()) if model_name(run) == model]
    return np.stack(cols, axis=1)


def bootstrap_gap_difference(tables, model_a="resnet_A", model_b="resnet_B", n_boot=2000, seed=0):
    """Paired bootstrap over test images of gap(A) - gap(B), gaps averaged over seeds."""
    gap_a = (correctness(tables, model_a, "original") - correctness(tables, model_a, "swapped")).mean(1)
    gap_b = (correctness(tables, model_b, "original") - correctness(tables, model_b, "swapped")).mean(1)
    diff = gap_a - gap_b
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {"gap_A": gap_a.mean(), "gap_B": gap_b.mean(), "difference": diff.mean(),
            "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
            "p_le_0": float((boots <= 0).mean())}


def bootstrap_accuracy_difference(tables, model_a, model_b, condition, n_boot=2000, seed=0):
    """Paired bootstrap over images of accuracy(B) - accuracy(A) in one condition, seeds averaged."""
    diff = correctness(tables, model_b, condition).mean(1) - correctness(tables, model_a, condition).mean(1)
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {"difference": diff.mean(), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)}


def per_breed_gap(tables, model):
    """Accuracy original - swapped per breed, averaged over seeds."""
    any_table = next(iter(tables.values()))
    gap = (correctness(tables, model, "original") - correctness(tables, model, "swapped")).mean(1)
    return pd.Series(gap, index=any_table.index).groupby(any_table["breed"]).mean()


def flip_examples(tables, flips_in="resnet_A", stable_in="resnet_B", n=8, seed=0):
    """Images that every seed of `flips_in` gets right on original and wrong on swapped,
    while every seed of `stable_in` gets right in both."""
    a_orig, a_swap = correctness(tables, flips_in, "original"), correctness(tables, flips_in, "swapped")
    b_orig, b_swap = correctness(tables, stable_in, "original"), correctness(tables, stable_in, "swapped")
    mask = (a_orig.min(1) == 1) & (a_swap.max(1) == 0) & (b_orig.min(1) == 1) & (b_swap.min(1) == 1)
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False)), int(mask.sum())
