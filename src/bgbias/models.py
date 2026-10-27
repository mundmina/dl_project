import torch
import torch.nn as nn
from torchvision import models as tvm

# arch -> (constructor, ImageNet weights, dotted path of the classification layer)
ARCHS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier.2"),
    "swin_t": (tvm.swin_t, tvm.Swin_T_Weights.IMAGENET1K_V1, "head"),
}


def build_model(arch, num_classes, pretrained=True):
    """torchvision model with its ImageNet head replaced by a `num_classes` linear layer."""
    ctor, weights, head = ARCHS[arch]
    model = ctor(weights=weights if pretrained else None)
    parent_name, _, child = head.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, nn.Linear(getattr(parent, child).in_features, num_classes))
    return model


def strip_head(model, arch):
    """Replace the classification layer with Identity in place: the model then returns penultimate features."""
    parent_name, _, child = ARCHS[arch][2].rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, nn.Identity())
    return model


def build_resnet18(num_classes, pretrained=True):
    return build_model("resnet18", num_classes, pretrained)


def load_checkpoint(path, num_classes, device):
    """Model from a best.pt written by train.py, plus the config it was trained with."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"].get("arch", "resnet18"), num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"]


def param_groups(model, lr_backbone, lr_head, arch="resnet18"):
    """Pretrained backbone gets a smaller learning rate than the new classification head."""
    head = list(model.get_submodule(ARCHS[arch][2]).parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]
