"""Stanford Dogs as an external test set: real photos of 21 of our dog breeds, taken by other people in
other places. No cut-and-paste, so this checks whether what we saw on synthetic backgrounds holds for a
natural distribution shift.

Two views per image: `full` (the whole photo) and `crop` (the annotated dog box, most of the background gone).

Caveat for the report: Stanford Dogs is built from ImageNet, so ImageNet-pretrained backbones have seen
these photos (with ImageNet's dog labels) during pretraining. That holds for A and B alike, so their
comparison stays fair; the scratch models never saw them.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from . import compose
from .data_setup import _fetch
from .utils import DRIVE_ROOT, LOCAL_DATA

SD_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
SD_ARCHIVES = ("images.tar", "annotation.tar")

# ImageNet synset -> Oxford-IIIT Pet breed. Pet breeds with no Stanford Dogs class are left out
# (american_bulldog, havanese, shiba_inu) and so is the pit bull (Stanford has only the AmStaff).
SYNSET_TO_PET = {
    "n02085620": "chihuahua",
    "n02085782": "japanese_chin",
    "n02088238": "basset_hound",
    "n02088364": "beagle",
    "n02093256": "staffordshire_bull_terrier",
    "n02094433": "yorkshire_terrier",
    "n02097298": "scottish_terrier",
    "n02098105": "wheaten_terrier",
    "n02100236": "german_shorthaired",
    "n02100735": "english_setter",
    "n02102318": "english_cocker_spaniel",
    "n02107312": "miniature_pinscher",
    "n02108089": "boxer",
    "n02109525": "saint_bernard",
    "n02110958": "pug",
    "n02111129": "leonberger",
    "n02111277": "newfoundland",
    "n02111500": "great_pyrenees",
    "n02111889": "samoyed",
    "n02112018": "pomeranian",
    "n02112350": "keeshond",
}


def prepare_stanford_dogs(data_dir=LOCAL_DATA, cache_dir=DRIVE_ROOT / "data"):
    root = Path(data_dir) / "stanford-dogs"
    _fetch(SD_URL, SD_ARCHIVES, root, Path(cache_dir) if cache_dir is not None else None)
    return root


def _dog_box(xml_path):
    """Union of all annotated dog boxes (some photos show several dogs)."""
    boxes = [[int(float(o.find("bndbox").find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]
             for o in ET.parse(xml_path).getroot().iter("object")]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def load_index(root, breed_to_label):
    """One row per photo of a mapped breed: path, breed, label (Pet label id), dog box."""
    root = Path(root)
    rows = []
    for folder in sorted((root / "Images").iterdir()):
        synset = folder.name.split("-", 1)[0]
        breed = SYNSET_TO_PET.get(synset)
        if breed is None:
            continue
        for img in sorted(folder.glob("*.jpg")):
            box = _dog_box(root / "Annotation" / folder.name / img.stem)
            rows.append({"path": str(img.relative_to(root)), "breed": breed, "label": breed_to_label[breed],
                         "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})
    return pd.DataFrame(rows)


class StanfordDogs(Dataset):
    def __init__(self, index, root, view="full", transform=None):
        assert view in ("full", "crop"), view
        self.df = index.reset_index(drop=True)
        self.root = Path(root)
        self.view = view
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def render(self, idx):
        r = self.df.iloc[idx]
        img = compose.load_image(self.root / r.path)
        if self.view == "crop":
            img = img.crop((max(0, r.x0), max(0, r.y0), min(img.width, r.x1), min(img.height, r.y1)))
        return img

    def __getitem__(self, idx):
        img = self.render(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.df.at[idx, "label"])


@torch.no_grad()
def predict(model, dataset, device, dog_labels, batch_size=128, num_workers=2):
    """Predicted Pet label over all 37 classes, and restricted to the dog breeds present here."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=device.type == "cuda")
    dog_labels = torch.as_tensor(sorted(dog_labels), device=device)
    model.eval()
    pred, pred_dog = [], []
    for x, _ in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True)).float()
        pred.append(logits.argmax(1).cpu())
        pred_dog.append(dog_labels[logits[:, dog_labels].argmax(1)].cpu())
    return torch.cat(pred).numpy(), torch.cat(pred_dog).numpy()
