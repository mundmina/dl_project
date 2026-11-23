"""How much of a decision comes from the pet and how much from the background.

Two attribution methods, both measured in the same unit: the log-probability of the true class,
f(x) = log softmax(model(x))[y].

1. Region Shapley values (two players: pet P and background B). Every image is shown in four versions
   that switch each region on or off; "off" means the neutral gray used for fg-only / bg-only:
       v(P+B) = f(full image)      v(P) = f(pet on gray)
       v(B)   = f(pet painted gray) v(0) = f(all gray)
   phi_P = 1/2 [(v(P+B) - v(B)) + (v(P) - v(0))]
   phi_B = 1/2 [(v(P+B) - v(P)) + (v(B) - v(0))]
   phi_P + phi_B = v(P+B) - v(0) exactly. Works for any architecture (forward passes only).

2. Integrated Gradients (Sundararajan et al., 2017), pixel level, baseline = the same all-gray image:
       IG_i(x) = (x_i - x'_i) * mean_{k=1..m} d f(x' + k/m (x - x')) / d x_i
   Summing IG over the pet mask and over the background gives the pixel-level split; the sum over all
   pixels approximates f(x) - f(x') (completeness), which we report as a check.

For a swapped image the "background" is the new Places365 scene, so phi_B on swapped images measures
how much the unfamiliar scene pushes the true class up or down.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from . import compose
from .gradcam import eval_mask_transform

# Order of the views returned by RegionViews.
VIEWS = ("original", "fgonly", "bgonly", "swapped", "swapped_bgonly", "gray")


class RegionViews(Dataset):
    """For each image: the six views as one tensor (6, 3, H, W), the pet mask (H, W) and the label.
    `base` is a PetBackgroundDataset with condition="swapped" and the eval transform (make_dataset(...,
    condition="swapped")); its fixed manifest gives the same new background as the swapped test condition."""

    def __init__(self, base, indices=None, img_size=224):
        self.base = base
        self.indices = list(range(len(base))) if indices is None else list(indices)
        self.mask_tf = eval_mask_transform(img_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        b = self.base
        image_id = b.df.at[i, "image_id"]
        img = compose.load_image(b.pet_root / "images" / f"{image_id}.jpg")
        alpha = b._alpha(image_id)
        swapped = compose.swap_background(img, alpha, b._background(b.bg_by_image[image_id]))
        views = {
            "original": img,
            "fgonly": compose.foreground_only(img, alpha),
            "bgonly": compose.background_only(img, alpha),
            "swapped": swapped,
            "swapped_bgonly": compose.background_only(swapped, alpha),
            "gray": compose.solid(img.size),
        }
        x = torch.stack([b.transform(views[v]) for v in VIEWS])
        mask = torch.from_numpy(
            np.asarray(self.mask_tf(Image.fromarray((alpha * 255).astype(np.uint8))), dtype=np.float32) / 255)
        return x, mask, int(b.df.at[i, "label"])


def true_logprob(model, x, y):
    """log p(y | x) for a batch."""
    return F.log_softmax(model(x).float(), dim=1).gather(1, y[:, None])[:, 0]


@torch.no_grad()
def region_shapley(model, views, y):
    """views: (N, 6, 3, H, W) in VIEWS order. Returns a dict of per-image tensors (log-prob units)."""
    n = views.shape[0]
    f = true_logprob(model, views.flatten(0, 1), y.repeat_interleave(len(VIEWS))).view(n, len(VIEWS))
    v = {name: f[:, j] for j, name in enumerate(VIEWS)}
    out = {}
    for cond, full, bg_only in (("original", "original", "bgonly"), ("swapped", "swapped", "swapped_bgonly")):
        phi_pet = 0.5 * ((v[full] - v[bg_only]) + (v["fgonly"] - v["gray"]))
        phi_bg = 0.5 * ((v[full] - v["fgonly"]) + (v[bg_only] - v["gray"]))
        out[cond] = {"phi_pet": phi_pet, "phi_bg": phi_bg, "f_full": v[full], "f_gray": v["gray"]}
    return out


def integrated_gradients(model, x, y, baseline, steps=32, chunk=64):
    """IG of log p(y|x) for ONE image x (3, H, W) against `baseline` (3, H, W).
    Returns the attribution map summed over colour channels (H, W) and the completeness error
    |sum IG - (f(x) - f(baseline))|."""
    alphas = (torch.arange(1, steps + 1, device=x.device, dtype=torch.float32) / steps)[:, None, None, None]
    path = baseline[None] + alphas * (x - baseline)[None]
    grads = torch.zeros_like(x)
    target = torch.full((chunk,), int(y), device=x.device, dtype=torch.long)
    for s in range(0, steps, chunk):
        p = path[s:s + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            out = true_logprob(model, p, target[:len(p)]).sum()
            g, = torch.autograd.grad(out, p)
        grads += g.sum(0)
    ig = ((x - baseline) * grads / steps).sum(0)
    with torch.no_grad():
        fx, fb = true_logprob(model, torch.stack([x, baseline]), target[:2]).tolist()
    return ig.detach(), abs(ig.sum().item() - (fx - fb))


def split_by_mask(attr, mask):
    """Signed sum on the pet and on the background, and the share of |attribution| on the pet."""
    pet = (attr * mask).sum().item()
    bg = (attr * (1 - mask)).sum().item()
    a = attr.abs()
    share = ((a * mask).sum() / a.sum().clamp_min(1e-12)).item()
    return pet, bg, share
