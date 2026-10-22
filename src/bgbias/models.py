import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes, pretrained=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path, num_classes, device):
    """Model from a best.pt written by train.py, plus the config it was trained with."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet18(num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"]


def param_groups(model, lr_backbone, lr_head):
    """Pretrained backbone gets a smaller learning rate than the new classification head."""
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]
