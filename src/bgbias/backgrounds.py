"""Background images from Places365 (val, 256px).

Scene categories are split into three disjoint pools (train / val / test), so the test
backgrounds come from scene types the model never saw during training. Categories where
animals are common are dropped up front.
"""
from pathlib import Path

import numpy as np
import pandas as pd

POOLS = ("train", "val", "test")

# Scene categories where animals regularly appear in the photo.
EXCLUDE_KEYWORDS = (
    "kennel", "pasture", "corral", "barn", "stable", "veterinarian", "pet_shop", "zoo",
    "aquarium", "ranch", "farm", "chicken", "cattle", "hunting", "fishpond", "field/wild",
    "rodeo", "racecourse",
)


def load_places_index(places_root):
    """All Places365 val images as (path relative to places_root, category)."""
    places_root = Path(places_root)
    idx_to_cat = {}
    with open(places_root / "categories_places365.txt") as fh:
        for line in fh:
            name, idx = line.split()
            idx_to_cat[int(idx)] = name
    rows = []
    with open(places_root / "places365_val.txt") as fh:
        for line in fh:
            fname, idx = line.split()
            rows.append({"bg_path": f"val_256/{fname}", "category": idx_to_cat[int(idx)]})
    return pd.DataFrame(rows)


def make_background_pools(index, fractions=(0.6, 0.15, 0.25), seed=42, exclude=EXCLUDE_KEYWORDS):
    """Assign every non-excluded category to exactly one pool."""
    categories = sorted(c for c in index["category"].unique() if not any(k in c for k in exclude))
    rng = np.random.default_rng(seed)
    categories = list(rng.permutation(categories))
    n_train = round(fractions[0] * len(categories))
    n_val = round(fractions[1] * len(categories))
    pool_of = {}
    for i, cat in enumerate(categories):
        pool_of[cat] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    pools = index[index["category"].isin(pool_of)].copy()
    pools["pool"] = pools["category"].map(pool_of)
    check_pools(pools)
    return pools.sort_values(["pool", "bg_path"]).reset_index(drop=True)


def check_pools(pools):
    cats = {p: set(pools.loc[pools["pool"] == p, "category"]) for p in POOLS}
    paths = {p: set(pools.loc[pools["pool"] == p, "bg_path"]) for p in POOLS}
    for a in POOLS:
        for b in POOLS:
            if a < b:
                assert not cats[a] & cats[b], f"pools {a} and {b} share categories"
                assert not paths[a] & paths[b], f"pools {a} and {b} share images"


def make_eval_manifest(split_df, pools, pool, seed):
    """Fixed background for every image of a val/test split, so all models see identical inputs."""
    candidates = pools[pools["pool"] == pool].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    replace = len(candidates) < len(split_df)
    picks = rng.choice(len(candidates), size=len(split_df), replace=replace)
    return pd.DataFrame({
        "image_id": split_df["image_id"].to_numpy(),
        "bg_path": candidates.loc[picks, "bg_path"].to_numpy(),
        "bg_category": candidates.loc[picks, "category"].to_numpy(),
    })
