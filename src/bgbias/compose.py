"""Mask handling and background replacement.

Trimap values: 1 = pet, 2 = background, 3 = border / uncertain.
"""
import numpy as np
from PIL import Image, ImageFilter

# Mean ImageNet colour, used as the neutral background.
NEUTRAL_RGB = (124, 116, 104)


def load_image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def load_trimap(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()


def foreground_alpha(trimap, include_border=True, blur_sigma=1.5):
    """Soft alpha mask in [0, 1]; 1 = pet. The border class keeps fur edges attached."""
    fg = trimap == 1
    if include_border:
        fg = fg | (trimap == 3)
    mask = Image.fromarray(fg.astype(np.uint8) * 255)
    if blur_sigma > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_sigma))
    return np.asarray(mask, dtype=np.float32) / 255.0


def fit_background(bg, size, offset=(0.5, 0.5)):
    """Scale `bg` to cover `size` (w, h), then crop. offset in [0, 1]^2 picks the crop position."""
    w, h = size
    scale = max(w / bg.width, h / bg.height)
    new_w, new_h = max(w, round(bg.width * scale)), max(h, round(bg.height * scale))
    bg = bg.resize((new_w, new_h), Image.BILINEAR)
    left = round((new_w - w) * offset[0])
    top = round((new_h - h) * offset[1])
    return bg.crop((left, top, left + w, top + h))


def composite(img, alpha, bg):
    """Pet from `img` over `bg` (already the same size as `img`)."""
    a = alpha[..., None]
    out = a * np.asarray(img, dtype=np.float32) + (1 - a) * np.asarray(bg, dtype=np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def solid(size, color=NEUTRAL_RGB):
    return Image.new("RGB", size, color)


def swap_background(img, alpha, bg, offset=(0.5, 0.5)):
    return composite(img, alpha, fit_background(bg, img.size, offset))


def foreground_only(img, alpha, color=NEUTRAL_RGB):
    return composite(img, alpha, solid(img.size, color))


def background_only(img, alpha, color=NEUTRAL_RGB):
    """Pet painted over with the neutral colour: can the model guess the breed from context alone?"""
    return composite(solid(img.size, color), alpha, img)
