"""Case study: Waterbirds (Sagawa et al., 2020), the standard benchmark for background shortcuts.

CUB birds pasted onto Places scenes. In train, 95% of waterbirds sit on water and 95% of landbirds on
land, so a model can score well by recognising the scene. The test set is balanced over the four
(bird, place) groups; the number that matters is worst-group accuracy (e.g. waterbird on land).

We apply our fix unchanged: swap the background with a random Places365 scene (our train pool) with
probability p. Waterbirds ships without masks, so they come from a pretrained segmentation network
(DeepLabV3, VOC class "bird"); no manual annotation is needed.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from . import compose
from .data_setup import _fetch
from .datasets import build_transform
from .models import build_model, param_groups
from .utils import DRIVE_ROOT, LOCAL_DATA, seed_everything

WB_URL = "https://nlp.stanford.edu/data/dro/"
WB_ARCHIVE = "waterbird_complete95_forest2water2.tar.gz"
SPLITS = {0: "train", 1: "val", 2: "test"}
GROUPS = {0: "landbird on land", 1: "landbird on water", 2: "waterbird on land", 3: "waterbird on water"}


def prepare_waterbirds(data_dir=LOCAL_DATA, cache_dir=DRIVE_ROOT / "data"):
    root = Path(data_dir) / "waterbirds"
    _fetch(WB_URL, (WB_ARCHIVE,), root, Path(cache_dir) if cache_dir is not None else None)
    return root / "waterbird_complete95_forest2water2"


def load_metadata(root):
    df = pd.read_csv(Path(root) / "metadata.csv")
    df["split"] = df["split"].map(SPLITS)
    df["group"] = 2 * df["y"] + df["place"]
    return df


@torch.no_grad()
def compute_masks(root, df, device, cache_path):
    """Bird mask per image from DeepLabV3 (VOC 'bird'), bit-packed in one .npz; reused when cached."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_masks(cache_path)
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    seg = deeplabv3_resnet50(weights=weights).to(device).eval()
    tf = weights.transforms()
    bird = weights.meta["categories"].index("bird")
    packed, t0 = {}, time.time()
    for i, r in enumerate(df.itertuples()):
        img = compose.load_image(Path(root) / r.img_filename)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            labels = seg(tf(img)[None].to(device))["out"][0].argmax(0)
        mask = Image.fromarray((labels == bird).cpu().numpy().astype(np.uint8) * 255).resize(img.size, Image.NEAREST)
        m = np.asarray(mask) > 127
        packed[str(r.img_id)] = np.concatenate([np.array(m.shape, dtype=np.uint32).view(np.uint8), np.packbits(m)])
        if i % 1000 == 0:
            print(f"{i}/{len(df)} masks, {time.time() - t0:.0f}s")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **packed)
    return load_masks(cache_path)


def load_masks(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def unpack_mask(packed):
    h, w = packed[:8].view(np.uint32)
    return np.unpackbits(packed[8:], count=int(h) * int(w)).reshape(int(h), int(w)).astype(bool)


class WaterbirdsDataset(Dataset):
    """Items: (image tensor, bird label 0/1, group 0..3). With p_swap > 0 the background is replaced by a
    random scene from `bg_pool` (paths under `places_root`), exactly as in our Pet training."""

    def __init__(self, df, root, transform, masks=None, p_swap=0.0, bg_pool=None, places_root=None, blur_sigma=1.5):
        if p_swap > 0:
            assert masks is not None and bg_pool and places_root is not None
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.masks = masks
        self.p_swap = p_swap
        self.bg_pool = list(bg_pool or [])
        self.places_root = Path(places_root) if places_root is not None else None
        self.blur_sigma = blur_sigma

    def __len__(self):
        return len(self.df)

    def alpha(self, idx):
        m = unpack_mask(self.masks[str(self.df.at[idx, "img_id"])])
        mask = Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(self.blur_sigma))
        return np.asarray(mask, dtype=np.float32) / 255

    def render(self, idx):
        img = compose.load_image(self.root / self.df.at[idx, "img_filename"])
        if self.p_swap > 0 and torch.rand(1).item() < self.p_swap:
            alpha = self.alpha(idx)
            if alpha.mean() > 0.005:   # segmentation found the bird; otherwise keep the photo
                bg = compose.load_image(self.places_root / self.bg_pool[torch.randint(len(self.bg_pool), (1,)).item()])
                return compose.swap_background(img, alpha, bg, tuple(torch.rand(2).tolist()))
        return img

    def __getitem__(self, idx):
        return self.transform(self.render(idx)), int(self.df.at[idx, "y"]), int(self.df.at[idx, "group"])


def _loader(ds, batch_size, shuffle, seed):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count(),
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys, groups = [], [], []
    for x, y, g in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        ys.append(y)
        groups.append(g)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy(), torch.cat(groups).numpy()


def group_metrics(pred, y, group):
    """Average accuracy, worst-group accuracy, and accuracy per group."""
    correct = pred == y
    per_group = {GROUPS[g]: float(correct[group == g].mean()) for g in sorted(GROUPS)}
    return {"avg_acc": float(correct.mean()), "worst_group_acc": min(per_group.values()), **per_group}


def train(cfg, seed, df, root, masks, places_root, bg_pool, run_dir, device):
    """ImageNet ResNet-18 fine-tuned on Waterbirds train; checkpoint chosen by val average accuracy
    (no group labels used for selection). Resumes from last.pt like the Pet training."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    split = {s: df[df.split == s] for s in SPLITS.values()}
    train_ds = WaterbirdsDataset(split["train"], root, build_transform(train=True), masks, cfg["p_swap"], bg_pool, places_root)
    train_loader = _loader(train_ds, cfg["batch_size"], True, seed)
    val_loader = _loader(WaterbirdsDataset(split["val"], root, build_transform(train=False)), 2 * cfg["batch_size"], False, seed)

    arch = cfg.get("arch", "resnet18")
    model = build_model(arch, 2).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg["lr_backbone"], cfg["lr_head"], arch), weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    start, best, history = 0, -1.0, []
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start, best, history = ckpt["epoch"] + 1, ckpt["best_acc"], ckpt["history"]
        print(f"resuming {run_dir.name} from epoch {start}")

    for epoch in range(start, cfg["epochs"]):
        train_loader.generator.manual_seed(seed * 1000 + epoch)
        model.train()
        t0 = time.time()
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        val = group_metrics(*predict(model, val_loader, device))
        row = {"epoch": epoch, "val_avg_acc": val["avg_acc"], "val_worst_group_acc": val["worst_group_acc"],
               "time_s": round(time.time() - t0, 1)}
        history.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        if val["avg_acc"] > best:
            best = val["avg_acc"]
            torch.save({"model": model.state_dict(), "config": cfg, "seed": seed, "epoch": epoch}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_acc": best, "history": history}, last_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    return history
