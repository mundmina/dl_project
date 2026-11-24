"""Training masks of different quality, for the mask-robustness experiment (paper plan, stage 1).

Model B needs a pet mask to paste the pet onto a new scene. Oxford-IIIT Pet has human trimaps, most
datasets do not. Here we train B with worse masks and check how much of the effect survives:

  deeplab - automatic: DeepLabV3 (COCO weights, VOC classes "cat" + "dog"), no annotation at all
  dilate  - human mask grown by r pixels: a halo of the original background travels with the pet
  erode   - human mask shrunk by r pixels: fur edges are cut off and replaced by the new scene
  box     - filled bounding box of the human mask: the crudest possible mask

r = frac * min(height, width) of the photo, so the error is the same relative size on every image.
Only the TRAINING composite uses these masks. Every test condition still uses the human trimaps, so the
test protocol and all test numbers stay comparable with the main experiment.

Masks are stored bit-packed in one .npz per variant (same format as the Waterbirds masks).
"""
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import compose

VARIANTS = ("deeplab", "dilate", "erode", "box")


def load_masks(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def unpack_mask(packed):
    h, w = packed[:8].view(np.uint32)
    return np.unpackbits(packed[8:], count=int(h) * int(w)).reshape(int(h), int(w)).astype(bool)


def pack_mask(m):
    m = np.asarray(m, dtype=bool)
    return np.concatenate([np.array(m.shape, dtype=np.uint32).view(np.uint8), np.packbits(m)])


def human_mask(pet_root, image_id, include_border=True):
    """Binary pet mask from the trimap, exactly as the main experiment uses it (before the soft blur)."""
    trimap = compose.load_trimap(Path(pet_root) / "annotations" / "trimaps" / f"{image_id}.png")
    fg = trimap == 1
    if include_border:
        fg = fg | (trimap == 3)
    return fg


def perturb(mask, variant, frac=0.03):
    """dilate / erode by r = frac * min(h, w) pixels (exact Euclidean disk), or the bounding box."""
    mask = np.asarray(mask, dtype=bool)
    if variant == "box":
        out = np.zeros_like(mask)
        ys, xs = np.nonzero(mask)
        if len(ys):
            out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        return out
    r = frac * min(mask.shape)
    if variant == "dilate":
        return distance_transform_edt(~mask) <= r
    if variant == "erode":
        return distance_transform_edt(mask) > r
    raise ValueError(variant)


def quality(mask, ref):
    """How a mask compares with the human one: IoU, share of the pet it covers (recall) and share of
    the mask that is actually pet (precision)."""
    mask, ref = np.asarray(mask, bool), np.asarray(ref, bool)
    inter = (mask & ref).sum()
    union = (mask | ref).sum()
    return {"iou": inter / max(union, 1), "recall": inter / max(ref.sum(), 1),
            "precision": inter / max(mask.sum(), 1), "area": mask.mean()}


def build_perturbed(pet_root, image_ids, variant, cache_path, frac=0.03, include_border=True):
    """Masks of one human-derived variant for `image_ids`, cached in `cache_path`."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_masks(cache_path)
    packed = {str(i): pack_mask(perturb(human_mask(pet_root, i, include_border), variant, frac)) for i in image_ids}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **packed)
    return load_masks(cache_path)


def build_deeplab(pet_root, image_ids, device, cache_path):
    """Automatic pet masks: DeepLabV3-ResNet50 (COCO weights with VOC labels), pixels labelled cat or dog."""
    import torch
    from PIL import Image
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_masks(cache_path)
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    seg = deeplabv3_resnet50(weights=weights).to(device).eval()
    tf = weights.transforms()
    cats = weights.meta["categories"]
    pet_classes = torch.tensor([cats.index("cat"), cats.index("dog")], device=device)
    packed, t0 = {}, time.time()
    with torch.no_grad():
        for k, image_id in enumerate(image_ids):
            img = compose.load_image(Path(pet_root) / "images" / f"{image_id}.jpg")
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                labels = seg(tf(img)[None].to(device))["out"][0].argmax(0)
            m = torch.isin(labels, pet_classes).cpu().numpy().astype(np.uint8) * 255
            m = np.asarray(Image.fromarray(m).resize(img.size, Image.NEAREST)) > 127
            packed[str(image_id)] = pack_mask(m)
            if k % 500 == 0:
                print(f"{k}/{len(image_ids)} masks, {time.time() - t0:.0f}s")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **packed)
    return load_masks(cache_path)


__all__ = ["VARIANTS", "pack_mask", "unpack_mask", "load_masks", "human_mask", "perturb", "quality",
           "build_perturbed", "build_deeplab"]
