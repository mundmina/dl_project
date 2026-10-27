"""PyTorch dataset for Oxford-IIIT Pet under the test conditions of the project.

condition:
  original  - unchanged image (training may still change the background with probability p_swap,
              how is set by bg_mode: "places" random scene, "gray" neutral colour, "blur" own background blurred)
  swapped   - pet on the fixed background from `manifest`
  fgonly    - pet on the neutral background
  bgonly    - pet painted over, background kept
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision import tv_tensors
from torchvision.transforms import v2

from . import compose

CONDITIONS = ("original", "swapped", "fgonly", "bgonly")
BG_MODES = ("places", "gray", "blur")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(train, size=224):
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), *norm])
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), *norm])


def build_joint_transform(size=224):
    """Train transform applied to the image and its pet mask together (same crop and flip)."""
    return v2.Compose([v2.ToImage(), v2.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
                       v2.RandomHorizontalFlip(), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class PetBackgroundDataset(Dataset):
    def __init__(self, split_df, pet_root, condition="original", transform=None,
                 places_root=None, manifest=None, bg_pool=None, p_swap=0.0,
                 include_border=True, blur_sigma=1.5, bg_mode="places", return_mask=False):
        """return_mask: items are (image, pet mask 1xHxW, label); `transform` must then take both
        (build_joint_transform). Used by the saliency-penalty training on original images."""
        assert condition in CONDITIONS, condition
        assert bg_mode in BG_MODES, bg_mode
        assert not return_mask or (condition == "original" and p_swap == 0)
        if condition == "swapped":
            assert manifest is not None and places_root is not None
        if p_swap > 0:
            assert condition == "original"
            if bg_mode == "places":
                assert bg_pool is not None and places_root is not None
        self.df = split_df.reset_index(drop=True)
        self.pet_root = Path(pet_root)
        self.places_root = Path(places_root) if places_root is not None else None
        self.condition = condition
        self.transform = transform
        self.bg_by_image = dict(zip(manifest["image_id"], manifest["bg_path"])) if manifest is not None else None
        self.bg_pool = list(bg_pool) if bg_pool is not None else []
        self.p_swap = p_swap
        self.bg_mode = bg_mode
        self.return_mask = return_mask
        self.include_border = include_border
        self.blur_sigma = blur_sigma

    def __len__(self):
        return len(self.df)

    def _alpha(self, image_id):
        trimap = compose.load_trimap(self.pet_root / "annotations" / "trimaps" / f"{image_id}.png")
        return compose.foreground_alpha(trimap, self.include_border, self.blur_sigma)

    def _background(self, rel_path):
        return compose.load_image(self.places_root / rel_path)

    def render(self, idx):
        """PIL image for item `idx` before the tensor transform. Randomness comes from torch,
        which DataLoader seeds per worker, so runs are reproducible given the loader generator."""
        image_id = self.df.at[idx, "image_id"]
        img = compose.load_image(self.pet_root / "images" / f"{image_id}.jpg")
        if self.condition == "original":
            if self.p_swap > 0 and torch.rand(1).item() < self.p_swap:
                if self.bg_mode == "gray":
                    return compose.foreground_only(img, self._alpha(image_id))
                if self.bg_mode == "blur":
                    return compose.blur_background(img, self._alpha(image_id))
                bg = self._background(self.bg_pool[torch.randint(len(self.bg_pool), (1,)).item()])
                offset = tuple(torch.rand(2).tolist())
                return compose.swap_background(img, self._alpha(image_id), bg, offset)
            return img
        alpha = self._alpha(image_id)
        if self.condition == "swapped":
            return compose.swap_background(img, alpha, self._background(self.bg_by_image[image_id]))
        if self.condition == "fgonly":
            return compose.foreground_only(img, alpha)
        return compose.background_only(img, alpha)

    def __getitem__(self, idx):
        img = self.render(idx)
        if self.return_mask:
            alpha = self._alpha(self.df.at[idx, "image_id"])
            img, mask = self.transform(img, tv_tensors.Mask(torch.from_numpy(alpha)[None]))
            return img, mask.as_subclass(torch.Tensor), int(self.df.at[idx, "label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.df.at[idx, "label"])
