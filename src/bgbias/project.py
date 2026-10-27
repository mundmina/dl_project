"""Everything a training / evaluation notebook needs: dataset paths, splits, background pools,
fixed eval manifests, and a single place that builds datasets from them."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_setup import prepare_pets, prepare_places
from .datasets import PetBackgroundDataset, build_joint_transform, build_transform
from .splits import load_splits
from .utils import DRIVE_ROOT

NUM_CLASSES = 37


@dataclass
class ProjectData:
    pet_root: Path
    places_root: Path
    splits: dict
    pools: pd.DataFrame
    manifests: dict
    num_classes: int = NUM_CLASSES

    def bg_pool(self, pool):
        return self.pools.loc[self.pools["pool"] == pool, "bg_path"].tolist()


def load_project_data(repo_dir, week1_dir=DRIVE_ROOT / "week1"):
    """Splits and manifests from the repo if committed, otherwise from the Drive copy made by 01_data."""
    for base in (Path(repo_dir), Path(week1_dir)):
        if (base / "splits" / "train.csv").exists() and (base / "manifests" / "test_swapped.csv").exists():
            break
    else:
        raise FileNotFoundError("splits / manifests not found: run notebooks/01_data.ipynb first")
    print("splits and manifests from", base)
    return ProjectData(
        pet_root=prepare_pets(),
        places_root=prepare_places(),
        splits=load_splits(base / "splits"),
        pools=pd.read_csv(base / "manifests" / "background_pools.csv"),
        manifests={s: pd.read_csv(base / "manifests" / f"{s}_swapped.csv") for s in ("val", "test")},
    )


def make_dataset(data, split, condition="original", train=False, p_swap=0.0, bg_mode="places",
                 include_border=True, blur_sigma=1.5, img_size=224, transform="auto", return_mask=False):
    """transform="auto" -> standard train/eval tensor transform; None -> PIL images (e.g. for HOG)."""
    if transform == "auto":
        transform = build_joint_transform(img_size) if return_mask else build_transform(train=train, size=img_size)
    return PetBackgroundDataset(
        data.splits[split], data.pet_root, condition=condition, transform=transform,
        places_root=data.places_root, manifest=data.manifests.get(split),
        bg_pool=data.bg_pool("train") if p_swap > 0 else None, p_swap=p_swap, bg_mode=bg_mode,
        include_border=include_border, blur_sigma=blur_sigma, return_mask=return_mask,
    )
