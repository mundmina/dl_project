"""Cue or distractor? A controlled background cue on Oxford-IIIT Pet (paper plan, new framing).

Two different ways a classifier can depend on the background:
  cue        - the background predicts the class and the model uses it (shortcut learning);
  distractor - the background carries no class information, yet an unfamiliar scene pushes the model off.
Our ResNet-18 A looks like a distractor case (bg-only ~ chance, foreign scene hurts). To show that the diagnostic
can tell the two apart, we CREATE a cue of known strength:

  every breed gets one "home" scene category from the training pool of Places365. When a training background is
  swapped (p = 0.5, as for model B), the new scene comes from the breed's home category with probability rho and
  from the random pool otherwise. rho = 0 is exactly model B; rho = 1 means every swapped image of a breed shows
  its home scene.

Each home category's 100 images are split: 80 for training, 20 held out for the test conditions below, so test
scenes are new images of the home category (the category is the cue, not the exact photo).

Test views per image (all with the human trimap, eval transform):
  original, fgonly, bgonly, gray                     - as in the main experiment
  swapped, swapped_bgonly                            - foreign scene from the test pool (the standard swap)
  congruent, congruent_bgonly                        - a held-out scene of the breed's own home category
  conflict, conflict_bgonly                          - a held-out scene of ANOTHER breed's home category
Diagnostic numbers (bgbias.cue.summarize):
  cue:        acc(congruent_bgonly) >> chance, phi_bg(congruent) > 0, conflict follow rate (predicts the breed
              whose home scene it is) >> 1/36
  distractor: phi_bg(foreign swapped) < 0 while acc(bgonly) ~ chance
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from . import compose
from .attribution import true_logprob
from .gradcam import eval_mask_transform

VIEWS = ("original", "fgonly", "bgonly", "gray", "swapped", "swapped_bgonly",
         "congruent", "congruent_bgonly", "conflict", "conflict_bgonly")
# (full view, bg-only view) pairs for the region Shapley split
SHAPLEY_PAIRS = {"own": ("original", "bgonly"), "foreign": ("swapped", "swapped_bgonly"),
                 "congruent": ("congruent", "congruent_bgonly"), "conflict": ("conflict", "conflict_bgonly")}


def make_home_scenes(pools, num_classes, seed=0, n_heldout=20):
    """Pick one training-pool scene category per class. Returns
    home_train {label: [paths]}, home_test {label: [paths]}, table (label, category, n_train, n_test)."""
    train_pool = pools[pools["pool"] == "train"]
    cats = sorted(train_pool["category"].unique())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(cats, size=num_classes, replace=False)
    home_train, home_test, rows = {}, {}, []
    for label, cat in enumerate(chosen):
        paths = sorted(train_pool.loc[train_pool["category"] == cat, "bg_path"])
        paths = list(rng.permutation(paths))
        home_test[label], home_train[label] = paths[:n_heldout], paths[n_heldout:]
        rows.append({"label": label, "category": cat, "n_train": len(home_train[label]), "n_test": n_heldout})
    return home_train, home_test, pd.DataFrame(rows)


def pools_without_heldout(pools, home_test):
    """The pools table minus the held-out home images, so the random training pool never shows them.
    Use as dataclasses.replace(data, pools=pools_without_heldout(data.pools, home_test))."""
    held = {p for paths in home_test.values() for p in paths}
    return pools[~pools["bg_path"].isin(held)].reset_index(drop=True)


def make_cue_manifest(split_df, home_test, seed=1):
    """Fixed congruent / conflict scene for every test image, identical for all models."""
    rng = np.random.default_rng(seed)
    labels = split_df["label"].to_numpy()
    n_cls = len(home_test)
    other = (labels + rng.integers(1, n_cls, size=len(labels))) % n_cls   # uniform over the other classes
    return pd.DataFrame({
        "image_id": split_df["image_id"].to_numpy(), "label": labels,
        "congruent_bg": [home_test[l][rng.integers(len(home_test[l]))] for l in labels],
        "conflict_label": other,
        "conflict_bg": [home_test[o][rng.integers(len(home_test[o]))] for o in other],
    })


class CueViews(Dataset):
    """Items: views (len(VIEWS), 3, H, W), pet mask (H, W), label, conflict label.
    `base` = make_dataset(data, "test", condition="swapped") (eval transform + the standard foreign manifest)."""

    def __init__(self, base, manifest, img_size=224):
        self.base = base
        self.m = manifest.set_index("image_id")
        self.mask_tf = eval_mask_transform(img_size)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        b = self.base
        image_id = b.df.at[i, "image_id"]
        img = compose.load_image(b.pet_root / "images" / f"{image_id}.jpg")
        alpha = b._alpha(image_id)
        row = self.m.loc[image_id]
        scenes = {"swapped": b._background(b.bg_by_image[image_id]),
                  "congruent": b._background(row["congruent_bg"]),
                  "conflict": b._background(row["conflict_bg"])}
        views = {"original": img, "fgonly": compose.foreground_only(img, alpha),
                 "bgonly": compose.background_only(img, alpha), "gray": compose.solid(img.size)}
        for name, scene in scenes.items():
            full = compose.swap_background(img, alpha, scene)
            views[name] = full
            views[f"{name}_bgonly"] = compose.background_only(full, alpha)
        x = torch.stack([b.transform(views[v]) for v in VIEWS])
        mask = torch.from_numpy(
            np.asarray(self.mask_tf(Image.fromarray((alpha * 255).astype(np.uint8))), dtype=np.float32) / 255)
        return x, mask, int(b.df.at[i, "label"]), int(row["conflict_label"])


@torch.no_grad()
def view_outputs(model, views, y, y_conflict):
    """For a batch: predicted class per view, log p(true) per view, log p(conflict class) on the conflict views."""
    n, v = views.shape[:2]
    logits = model(views.flatten(0, 1)).float()
    logp = F.log_softmax(logits, dim=1).view(n, v, -1)
    pred = logp.argmax(-1)
    lp_true = logp.gather(2, y[:, None, None].expand(n, v, 1))[..., 0]
    lp_conf = logp.gather(2, y_conflict[:, None, None].expand(n, v, 1))[..., 0]
    return pred, lp_true, lp_conf


def per_image_table(pred, lp_true, lp_conf, y, y_conflict):
    """One row per image: correctness per view, Shapley phi_pet / phi_bg per pair, conflict follow."""
    out = {f"correct_{v}": (pred[:, j] == y).numpy() for j, v in enumerate(VIEWS)}
    f = {v: lp_true[:, j].numpy() for j, v in enumerate(VIEWS)}
    for name, (full, bg_only) in SHAPLEY_PAIRS.items():
        out[f"phi_pet_{name}"] = 0.5 * ((f[full] - f[bg_only]) + (f["fgonly"] - f["gray"]))
        out[f"phi_bg_{name}"] = 0.5 * ((f[full] - f["fgonly"]) + (f[bg_only] - f["gray"]))
    j = VIEWS.index("conflict")
    out["follow_conflict"] = (pred[:, j] == y_conflict).numpy()
    out["follow_conflict_bgonly"] = (pred[:, VIEWS.index("conflict_bgonly")] == y_conflict).numpy()
    out["logp_conflict_class"] = lp_conf[:, j].numpy()
    return pd.DataFrame(out)


def summarize(table, chance=1 / 37):
    """Diagnostic numbers for one run from its per-image table."""
    s = {f"acc_{v}": table[f"correct_{v}"].mean() for v in VIEWS if v != "gray"}
    s["gap_foreign"] = s["acc_original"] - s["acc_swapped"]
    s["gap_conflict"] = s["acc_original"] - s["acc_conflict"]
    s["follow_conflict"] = table["follow_conflict"].mean()
    s["follow_conflict_bgonly"] = table["follow_conflict_bgonly"].mean()
    for name in SHAPLEY_PAIRS:
        s[f"phi_bg_{name}"] = table[f"phi_bg_{name}"].mean()
        s[f"phi_pet_{name}"] = table[f"phi_pet_{name}"].mean()
    # the two axes of the diagnostic
    s["cue_score"] = s["acc_congruent_bgonly"] - chance          # can the home scene alone name the breed?
    s["distraction_score"] = -s["phi_bg_foreign"]                # how much a foreign scene pushes away (nats)
    return s
