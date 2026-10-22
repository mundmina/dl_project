"""Download Oxford-IIIT Pet and Places365 (val, 256px), caching archives on Google Drive.

The first session downloads from the original servers and copies the archives to Drive.
Later sessions copy the archives back from Drive, which is faster and does not depend on
the Oxford / MIT servers being up.

Each archive gets a `.<name>.done` marker after a complete extraction, so an interrupted
download or extraction is redone on the next run instead of being treated as finished.
"""
import shutil
import tarfile
import urllib.request
from pathlib import Path

from .utils import DRIVE_ROOT, LOCAL_DATA, pet_root, places_root

PET_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
PET_ARCHIVES = ("images.tar.gz", "annotations.tar.gz")
PLACES_URL = "http://data.csail.mit.edu/places/places365/"
PLACES_ARCHIVES = ("filelist_places365-standard.tar", "val_256.tar")


def _download(url, dst):
    part = dst.with_name(dst.name + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    part.rename(dst)


def _extract(archive, root):
    print(f"extracting {archive.name}")
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(root, filter="data")
        else:
            tar.extractall(root)


def _fetch(base_url, names, root, cache_dir):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        marker = root / f".{name}.done"
        if marker.exists():
            continue
        archive = root / name
        cached = cache_dir / name if cache_dir is not None else None
        if not archive.exists() and cached is not None and cached.exists():
            print(f"copying {name} from Drive cache")
            shutil.copy(cached, archive)
        if not archive.exists():
            _download(base_url + name, archive)
        try:
            _extract(archive, root)
        except (tarfile.TarError, EOFError, OSError):
            print(f"{name} is broken, downloading again")
            archive.unlink()
            _download(base_url + name, archive)
            _extract(archive, root)
        marker.touch()
        if cache_dir is not None and not (cached.exists() and cached.stat().st_size == archive.stat().st_size):
            print(f"caching {name} to Drive")
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(archive, cached)


def prepare_pets(data_dir=LOCAL_DATA, cache_dir=DRIVE_ROOT / "data"):
    root = pet_root(data_dir)
    _fetch(PET_URL, PET_ARCHIVES, root, Path(cache_dir) if cache_dir is not None else None)
    return root


def prepare_places(data_dir=LOCAL_DATA, cache_dir=DRIVE_ROOT / "data"):
    root = places_root(data_dir)
    _fetch(PLACES_URL, PLACES_ARCHIVES, root, Path(cache_dir) if cache_dir is not None else None)
    return root
