"""Train / val / test splits.

test  = official Oxford-IIIT Pet test split, untouched until final evaluation.
val   = stratified (by breed) fraction of the official trainval split.
train = the rest of trainval.
Splits are made on original image ids, before any background variants exist.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "val", "test")
SPECIES = {1: "cat", 2: "dog"}


def read_official_split(pet_root, split):
    """Parse annotations/{trainval,test}.txt: `<image_id> <class 1..37> <species 1|2> <breed id>`."""
    rows = []
    with open(Path(pet_root) / "annotations" / f"{split}.txt") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            image_id, class_id, species, _ = line.split()
            rows.append({
                "image_id": image_id,
                "label": int(class_id) - 1,
                "breed": image_id.rsplit("_", 1)[0].lower(),
                "species": SPECIES[int(species)],
            })
    return pd.DataFrame(rows)


def make_splits(pet_root, val_frac=0.15, seed=42):
    trainval = read_official_split(pet_root, "trainval")
    test = read_official_split(pet_root, "test")
    train, val = train_test_split(
        trainval, test_size=val_frac, stratify=trainval["label"], random_state=seed
    )
    splits = {
        "train": train.sort_values("image_id").reset_index(drop=True),
        "val": val.sort_values("image_id").reset_index(drop=True),
        "test": test.sort_values("image_id").reset_index(drop=True),
    }
    check_splits(splits)
    return splits


def check_splits(splits):
    ids = {name: set(df["image_id"]) for name, df in splits.items()}
    for a in SPLIT_NAMES:
        assert len(ids[a]) == len(splits[a]), f"duplicate image ids in {a}"
        for b in SPLIT_NAMES:
            if a < b:
                assert not ids[a] & ids[b], f"{a} and {b} overlap"
    n_classes = splits["train"]["label"].nunique()
    for name, df in splits.items():
        assert df["label"].nunique() == n_classes, f"{name} is missing classes"


def save_splits(splits, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


def load_splits(splits_dir):
    splits = {name: pd.read_csv(Path(splits_dir) / f"{name}.csv") for name in SPLIT_NAMES}
    check_splits(splits)
    return splits


def class_table(splits):
    """Images per breed in each split."""
    counts = {name: df.groupby(["label", "breed", "species"]).size() for name, df in splits.items()}
    return pd.DataFrame(counts).reset_index()


def audit_images(pet_root, image_ids):
    """Per image: size, colour mode, whether image/trimap load, and trimap class fractions."""
    pet_root = Path(pet_root)
    rows = []
    for image_id in image_ids:
        row = {"image_id": image_id}
        try:
            with Image.open(pet_root / "images" / f"{image_id}.jpg") as im:
                row.update(width=im.width, height=im.height, mode=im.mode)
                im.convert("RGB").load()
            row["image_ok"] = True
        except Exception as e:  # noqa: BLE001 - we want to record any failure
            row.update(image_ok=False, error=str(e))
        try:
            trimap = np.asarray(Image.open(pet_root / "annotations" / "trimaps" / f"{image_id}.png"))
            row["trimap_ok"] = trimap.shape[:2] == (row.get("height"), row.get("width"))
            for value, name in ((1, "fg_frac"), (2, "bg_frac"), (3, "border_frac")):
                row[name] = float((trimap == value).mean())
        except Exception as e:  # noqa: BLE001
            row.update(trimap_ok=False, error=str(e))
        rows.append(row)
    return pd.DataFrame(rows)
