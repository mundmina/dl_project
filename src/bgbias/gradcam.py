"""Grad-CAM for ResNet-18 (last conv block) and a measurement built on it:
how much of the CAM mass falls on the pet (trimap foreground) rather than the background."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from .project import make_dataset


class GradCAM:
    def __init__(self, model, layer=None):
        self.model = model
        self.acts = self.grads = None
        layer = layer if layer is not None else model.layer4
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda grad: setattr(self, "grads", grad))

    def remove(self):
        self._handle.remove()

    def __call__(self, x, target=None):
        """CAM in [0, 1] per image, same size as x; target defaults to the predicted class."""
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(x)
            if target is None:
                target = logits.argmax(1)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, target[:, None]).sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / cam.flatten(1).max(1).values.clamp_min(1e-8)[:, None, None]
        return cam.detach(), target.detach()


def eval_mask_transform(img_size=224):
    """Same geometry as the eval image transform, applied to the alpha mask."""
    return T.Compose([T.Resize(int(img_size * 256 / 224)), T.CenterCrop(img_size)])


def foreground_masks(dataset, indices, img_size=224):
    tf = eval_mask_transform(img_size)
    masks = []
    for i in indices:
        alpha = dataset._alpha(dataset.df.at[i, "image_id"])
        masks.append(np.asarray(tf(Image.fromarray((alpha * 255).astype(np.uint8))), dtype=np.float32) / 255)
    return torch.from_numpy(np.stack(masks))


def foreground_share(model, data, split, condition, indices, device, batch_size=32, **dataset_kwargs):
    """Per image: share of Grad-CAM mass on the pet. Also returns the pet's share of the image area."""
    ds = make_dataset(data, split, condition=condition, **dataset_kwargs)
    cam = GradCAM(model)
    shares, areas = [], []
    for start in range(0, len(indices), batch_size):
        idx = list(indices[start:start + batch_size])
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        masks = foreground_masks(ds, idx, x.shape[-1]).to(device)
        c, _ = cam(x)
        shares.append(((c * masks).flatten(1).sum(1) / c.flatten(1).sum(1).clamp_min(1e-8)).cpu())
        areas.append(masks.flatten(1).mean(1).cpu())
    cam.remove()
    return torch.cat(shares).numpy(), torch.cat(areas).numpy()
