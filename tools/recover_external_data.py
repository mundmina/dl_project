"""Recover exact external datasets from mirrors and rerun the failed notebooks."""
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path("C:/bgbias_drive")
CACHE = ROOT / "data"
LOCAL = Path("C:/bgbias_data")
REPO = Path(__file__).resolve().parents[1]
ARCHIVE = "https://web.archive.org/web/20250724000413id_/http://vision.stanford.edu/aditya86/ImageNetDogs/"
WB_URL = "https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"


def fetch(name, url, expected=None, archive_type="tar"):
    target = CACHE / name
    if target.exists() and (expected is None or hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == expected):
        print("Reusing", target, flush=True)
        return target
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(3):
        try:
            print("Downloading", name, "attempt", attempt + 1, flush=True)
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    count += len(block)
                    if count % (32 * 1024 * 1024) == 0:
                        print(name, count // (1024 * 1024), "MiB", flush=True)
            if expected and digest.hexdigest() != expected:
                raise ValueError("Checksum mismatch: " + name)
            if archive_type == "tar":
                with tarfile.open(partial) as archive:
                    if not archive.getmembers():
                        raise ValueError("Empty archive")
            elif archive_type == "zip":
                with zipfile.ZipFile(partial) as archive:
                    if not archive.namelist():
                        raise ValueError("Empty ZIP")
            partial.replace(target)
            print("Verified archive", name, count, "bytes, SHA256", digest.hexdigest(), flush=True)
            return target
        except Exception as error:
            print(name, type(error).__name__, str(error), flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def dogs():
    hashes = {"images.tar": "a443fc2f9a851aa1a7b8e4fa4af94b8459fce2c9f5a2f4dd4669cf9673a4ec58",
              "annotation.tar": "662ed36f707be6b195c2fda392d8e153e83af28cedc8e2134a649f198def30fd"}
    root = LOCAL / "stanford-dogs"
    root.mkdir(parents=True, exist_ok=True)
    archive = fetch("stanford_dogs_kaggle.zip", "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/stanford-dogs-dataset", archive_type="zip")
    with zipfile.ZipFile(archive) as contents:
        for member in contents.infolist():
            parts = Path(member.filename).parts
            for folder in ("Images", "Annotation"):
                if folder in parts and not member.is_dir():
                    relative = Path(*parts[parts.index(folder):])
                    target = (root / relative).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError("Unsafe ZIP path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with contents.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
    images = list((root / "Images").glob("*/*.jpg"))
    assert len(images) == 20580, len(images)
    assert all((root / "Annotation" / p.parent.name / p.stem).exists() for p in images)
    import xml.etree.ElementTree as ET
    for p in images:
        annotation = root / "Annotation" / p.parent.name / p.stem
        assert ET.parse(annotation).getroot().find("object/bndbox") is not None
    for name in hashes:
        (root / ("." + name + ".done")).touch()
    print("Stanford Dogs validated: 20,580 images and matching annotations", flush=True)


def waterbirds():
    import pyarrow.parquet as pq
    root = LOCAL / "waterbirds" / "waterbird_complete95_forest2water2"
    root.mkdir(parents=True, exist_ok=True)
    metadata_url = "https://huggingface.co/datasets/arubique/waterbirds/resolve/main/metadata.csv"
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        (root / "metadata.csv").write_bytes(response.read())
    with (root / "metadata.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_file = {Path(r["img_filename"]).name: r for r in rows}
    assert len(by_file) == len(rows)
    files = {"train": "575b462a17ae00e6a03b4f1833e3a5952e28492e278d3246186586c27a46367a",
             "validation": "9ab73e8625fb5856e132ecbbe509f20829e5d59a9bd087b6cda86f5c478350fb",
             "test": "fdb8696ce4db78c0ad48dd0120aaae40ce4330416d37fc41c9ea57faadc007f7"}
    seen = set()
    for split_id, (split, checksum) in enumerate(files.items()):
        name = split + "-00000-of-00001.parquet"
        archive = fetch("waterbirds_" + name,
                        "https://huggingface.co/datasets/grodino/waterbirds/resolve/e9856c710d0da2e4029d116cdd9d5fce7cc2bc80/data/" + name,
                        checksum, archive_type="parquet")
        for item in pq.read_table(archive).to_pylist():
            filename = Path(item["image"]["path"]).name
            row = by_file[filename]
            assert int(row["split"]) == split_id
            assert int(row["y"]) == item["label"] and int(row["place"]) == item["place"]
            target = (root / row["img_filename"]).resolve()
            assert target.is_relative_to(root.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item["image"]["bytes"])
            seen.add(filename)
    assert len(rows) == 11788, len(rows)
    assert len(seen) == 11788, len(seen)
    assert all((root / row["img_filename"]).is_file() for row in rows)
    splits = {str(i): sum(row["split"] == str(i) for row in rows) for i in range(3)}
    assert splits == {"0": 4795, "1": 1199, "2": 5794}, splits
    (LOCAL / "waterbirds" / ".waterbird_complete95_forest2water2.tar.gz.done").touch()
    print("Waterbirds validated: 11,788 images; official splits", splits, flush=True)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    success = []
    if "--resume" in sys.argv:
        required = [LOCAL / "stanford-dogs" / ".images.tar.done",
                    LOCAL / "stanford-dogs" / ".annotation.tar.done",
                    LOCAL / "waterbirds" / ".waterbird_complete95_forest2water2.tar.gz.done"]
        for marker in required:
            if not marker.exists():
                raise FileNotFoundError(f"Validated dataset marker missing: {marker}")
        success = ["04b_external_stanford_dogs", "07_case_waterbirds"]
        print("Resuming validated local datasets; existing prediction caches will be reused", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks = {pool.submit(dogs): "04b_external_stanford_dogs", pool.submit(waterbirds): "07_case_waterbirds"}
            for future in concurrent.futures.as_completed(tasks):
                try:
                    future.result()
                    success.append(tasks[future])
                except Exception as error:
                    print("RECOVERY FAILED", tasks[future], repr(error), flush=True)
    if not success:
        raise RuntimeError("No datasets recovered")
    spec = importlib.util.spec_from_file_location("local_batch", REPO / "tools" / "run_local_batch.py")
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    batch.QUEUE = [n for n in ["04b_external_stanford_dogs", "07_case_waterbirds"] if n in success]
    batch.STATE = json.loads((batch.OUT / "status.json").read_text())
    batch.STATE.pop("finished", None)
    batch.STATE.pop("paused_at", None)
    batch.STATE["state"] = "running"
    for name in batch.QUEUE:
        batch.STATE["notebooks"][name] = {"state": "queued"}
    batch.STATE["pid"] = os.getpid()
    batch.STATE["recovery_log"] = str(ROOT / "executed" / "remaining_batch" / "recovery.log")
    batch.main()


if __name__ == "__main__":
    main()
