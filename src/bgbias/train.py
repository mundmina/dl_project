"""Training. Within one architecture, A and B share everything except `p_swap` in the config.
Optional config keys: arch (default resnet18), pretrained (default true), bg_mode (default places),
cutmix (probability of applying CutMix to a batch, default 0),
rrr_lambda (weight of the saliency penalty, default 0; "right for the right reasons", Ross et al. 2017).

Saliency penalty: input gradient of the summed log-probabilities, squared and summed over channels;
the penalty is the share of that gradient energy that falls on the background (by the trimap mask).
It lies in [0, 1], so rrr_lambda is comparable to the cross-entropy. These runs use fp32: the
double backward through fp16 input gradients underflows.

Checkpoints go to `run_dir` (on Drive) every epoch; re-running `train` with the same
run_dir resumes from `last.pt`. The loader generator is reseeded every epoch from
(seed, epoch), so shuffling and augmentation do not depend on where a run was resumed.
"""
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .models import build_model, param_groups
from .project import make_dataset
from .utils import seed_everything


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def make_loader(dataset, batch_size, shuffle, seed, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def _save(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def background_share(model_out, x, mask):
    """Per image: share of the squared input-gradient energy on the background. Differentiable."""
    logp = model_out.float().log_softmax(1).sum()
    (grad,) = torch.autograd.grad(logp, x, create_graph=True)
    energy = grad.pow(2).sum(1, keepdim=True)
    return (energy * (1 - mask)).flatten(1).sum(1) / energy.flatten(1).sum(1).clamp_min(1e-12)


def train(cfg, seed, data, run_dir, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    common = dict(include_border=cfg["include_border"], blur_sigma=cfg["blur_sigma"], img_size=cfg["img_size"])
    workers = cfg.get("num_workers", os.cpu_count())
    rrr = cfg.get("rrr_lambda", 0.0)
    assert not (rrr > 0 and cfg.get("cutmix", 0.0) > 0), "saliency penalty and CutMix are not combined"
    use_amp = device.type == "cuda" and rrr == 0
    train_loader = make_loader(make_dataset(data, "train", train=True, p_swap=cfg["p_swap"],
                                            bg_mode=cfg.get("bg_mode", "places"), return_mask=rrr > 0, **common),
                               cfg["batch_size"], True, seed, workers)
    # val swapped is for monitoring only; checkpoints are selected on val original.
    val_loaders = {c: make_loader(make_dataset(data, "val", condition=c, **common),
                                  2 * cfg["batch_size"], False, seed, workers)
                   for c in ("original", "swapped")}

    arch = cfg.get("arch", "resnet18")
    model = build_model(arch, data.num_classes, pretrained=cfg.get("pretrained", True)).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg["lr_backbone"], cfg["lr_head"], arch),
                                  weight_decay=cfg["weight_decay"])
    p_cutmix = cfg.get("cutmix", 0.0)
    cutmix = v2.CutMix(num_classes=data.num_classes)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, best_acc, history = 0, -1.0, []
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_acc, history = ckpt["epoch"] + 1, ckpt["best_acc"], ckpt["history"]
        print(f"resuming {run_dir.name} from epoch {start_epoch}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}))

    for epoch in range(start_epoch, cfg["epochs"]):
        train_loader.generator.manual_seed(seed * 1000 + epoch)
        model.train()
        t0 = time.time()
        loss_sum = correct = total = 0
        bg_share_sum = 0.0
        for batch in train_loader:
            if rrr > 0:
                x, mask, y = batch
                mask = mask.to(device, non_blocking=True)
            else:
                x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            target = y
            if p_cutmix > 0 and torch.rand(1).item() < p_cutmix:
                x, target = cutmix(x, y)   # soft labels; train_acc below still counts the hard label
            if rrr > 0:
                x.requires_grad_(True)
            with torch.autocast(device.type, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out.float(), target)
            if rrr > 0:
                share = background_share(out, x, mask)
                loss = loss + rrr * share.mean()
                bg_share_sum += share.sum().item()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)

        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total}
        if rrr > 0:
            row["train_bg_share"] = bg_share_sum / total
        row.update({f"val_acc_{c}": accuracy(model, loader, device) for c, loader in val_loaders.items()})
        row["time_s"] = round(time.time() - t0, 1)
        history.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

        if row["val_acc_original"] > best_acc:
            best_acc = row["val_acc_original"]
            _save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_acc, "config": cfg, "seed": seed},
                  best_path)
        _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
               "epoch": epoch, "best_acc": best_acc, "history": history}, last_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    return history
