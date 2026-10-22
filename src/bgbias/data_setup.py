"""Download Oxford-IIIT Pet and Places365 (val, 256px), caching archives on Google Drive.

The first session downloads from the original servers and copies the archives to Drive.
Later sessions copy the archives back from Drive, which is faster and does not depend on
the Oxford / MIT servers being up. torchvision skips the download when an archive with
the right md5 is already in place and only extracts it.
"""
import shutil
from pathlib import Path

from .utils import DRIVE_ROOT, LOCAL_DATA, pet_root, places_root

PET_ARCHIVES = ("images.tar.gz", "annotations.tar.gz")
PLACES_ARCHIVES = ("filelist_places365-standard.tar", "val_256.tar")


def _restore(names, archive_dir, cache_dir):
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src, dst = cache_dir / name, archive_dir / name
        if src.exists() and not dst.exists():
            print(f"copying {name} from Drive cache")
            shutil.copy(src, dst)


def _save_to_cache(names, archive_dir, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src, dst = archive_dir / name, cache_dir / name
        if src.exists() and not dst.exists():
            print(f"caching {name} to Drive")
            shutil.copy(src, dst)


def prepare_pets(data_dir=LOCAL_DATA, cache_dir=DRIVE_ROOT / "data"):
    from torchvision.datasets import OxfordIIITPet

    root = pet_root(data_dir)
    if cache_dir is not None:
        _restore(PET_ARCHIVES, root, Path(cache_dir))
    OxfordIIITPet(root=str(data_dir), split="trainval", download=True)
    if cache_dir is not None:
        _save_to_cache(PET_ARCHIVES, root, Path(cache_dir))
    return root


def prepare_places(data_dir=LOCAL_DATA, cache_dir=DRIVE_ROOT / "data"):
    from torchvision.datasets import Places365

    root = places_root(data_dir)
    if cache_dir is not None:
        _restore(PLACES_ARCHIVES, root, Path(cache_dir))
    # Places365 refuses to download when the image folder already exists.
    download = not (root / "val_256").exists()
    Places365(root=str(root), split="val", small=True, download=download)
    if cache_dir is not None:
        _save_to_cache(PLACES_ARCHIVES, root, Path(cache_dir))
    return root
