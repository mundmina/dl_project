"""Generate the Colab notebooks in notebooks/ (plain JSON, no ML imports).

Usage: python tools/make_notebooks.py notebooks
Edit notebooks here, not in Colab, so they stay in sync with src/."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def colab_badge(name):
    url = f"https://colab.research.google.com/github/mundmina/dl_project/blob/main/notebooks/{name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


SETUP = code('''
# Shared setup: Drive, repo, dependencies. Same cell in every notebook.
# Colab: nothing to set. Local machine (own GPU): see LOCAL.md — open this notebook from the repo's notebooks/
# folder and set BGBIAS_DRIVE, BGBIAS_DATA and GITHUB_TOKEN before starting Jupyter.
REPO_URL = "https://github.com/mundmina/dl_project.git"
BRANCH = "main"

import os, subprocess, sys
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    REPO_DIR = "/content/dl_project"
    drive.mount("/content/drive")
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
    else:  # always run the latest code from GitHub
        subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "checkout", "-B", BRANCH, f"origin/{BRANCH}"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt"], check=True)
else:
    REPO_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))   # notebook opened from <repo>/notebooks
    assert os.path.exists(os.path.join(REPO_DIR, "src", "bgbias")), "open the notebook from the repo's notebooks/ folder"
    missing = [v for v in ("BGBIAS_DRIVE", "BGBIAS_DATA") if not os.environ.get(v)]
    assert not missing, f"set {missing} before starting Jupyter (see LOCAL.md)"
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH], check=True)
commit = subprocess.run(["git", "-C", REPO_DIR, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout
print("code version:", commit)
if f"{REPO_DIR}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_DIR}/src")
# DataLoader workers on Windows start fresh interpreters: they find bgbias through PYTHONPATH, not sys.path
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (f"{REPO_DIR}/src", os.environ.get("PYTHONPATH")) if p)
try:  # ask for secret access now, not at the end of a long run (the prompt times out unattended)
    if not os.environ.get("GITHUB_TOKEN"):
        from google.colab import userdata
        userdata.get("GITHUB_TOKEN")
    print("GITHUB_TOKEN: ok")
except Exception as e:  # noqa: BLE001
    print(f"GITHUB_TOKEN not available ({type(e).__name__}): results will not be pushed to GitHub")
''')

DATA = code('''
from bgbias.data_setup import prepare_pets, prepare_places
from bgbias.utils import DRIVE_ROOT, LOCAL_DATA

PET_ROOT = prepare_pets()        # /content/data/oxford-iiit-pet
PLACES_ROOT = prepare_places()   # /content/data/places365
OUT_DRIVE = DRIVE_ROOT / "week1"
OUT_DRIVE.mkdir(parents=True, exist_ok=True)
print(PET_ROOT, PLACES_ROOT)
''')

setup_nb = nb([
    md(f'''
# 00 · Setup
{colab_badge("00_setup.ipynb")}

Проверка окружения. **Runtime → Change runtime type → T4 GPU**, затем *Run all*.

Первый запуск скачивает Oxford-IIIT Pet (~800 MB) и Places365 val 256px (~500 MB) и кладёт архивы в
`MyDrive/bgbias/data/`. Следующие запуски берут архивы из Drive.
'''),
    SETUP,
    code('''
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),
    DATA,
    code('''
from pathlib import Path
n_img = len(list((PET_ROOT / "images").glob("*.jpg")))
n_tri = len([p for p in (PET_ROOT / "annotations" / "trimaps").glob("*.png") if not p.name.startswith("._")])
n_bg = len(list((PLACES_ROOT / "val_256").glob("*.jpg")))
print(f"pet images: {n_img}  trimaps: {n_tri}  places backgrounds: {n_bg}")
assert n_img > 7000 and n_tri > 7000 and n_bg == 36500
'''),
    code('''
import matplotlib.pyplot as plt
from bgbias import compose

image_id = "Abyssinian_1"
img = compose.load_image(PET_ROOT / "images" / f"{image_id}.jpg")
tri = compose.load_trimap(PET_ROOT / "annotations" / "trimaps" / f"{image_id}.png")
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img); ax[1].imshow(tri, cmap="viridis")
for a in ax: a.axis("off")
plt.show()
'''),
    md("Если всё прошло без ошибок — окружение готово, переходим к `01_data.ipynb`."),
])

data_nb = nb([
    md(f'''
# 01 · Data: splits, masks, backgrounds
{colab_badge("01_data.ipynb")}

Неделя 1 из `PLAN.md`:
1. Stratified train/val split из `trainval`, официальный `test` без изменений.
2. Аудит изображений и масок.
3. Пулы фонов из Places365 (train / val / test — разные категории сцен).
4. Фиксированные фоны для val и test (manifest).
5. Галерея «оригинал → маска → новый фон → нейтральный фон → только фон».
6. Smoke test DataLoader с заменой фона p = 0.5.

Результаты сохраняются в репозиторий (`splits/`, `manifests/`, `results/`, `report/figures/`) и в `MyDrive/bgbias/week1/`.
'''),
    SETUP,
    DATA,
    md("## 1. Splits"),
    code('''
from pathlib import Path
from bgbias import splits as S

SEED = 42
VAL_FRAC = 0.15
splits = S.make_splits(PET_ROOT, val_frac=VAL_FRAC, seed=SEED)
S.save_splits(splits, Path(REPO_DIR) / "splits")
for name, df in splits.items():
    print(f"{name:5s} {len(df):5d} images | {df.label.nunique()} breeds | "
          f"cats {(df.species == 'cat').sum()}, dogs {(df.species == 'dog').sum()}")
'''),
    code('''
table = S.class_table(splits)
table["total"] = table[["train", "val", "test"]].sum(axis=1)
print("min / max images per breed in train:", table.train.min(), table.train.max())
table
'''),
    md("## 2. Аудит изображений и масок"),
    code('''
import pandas as pd
all_ids = pd.concat(splits.values()).image_id
audit = S.audit_images(PET_ROOT, all_ids)
audit.to_csv(Path(REPO_DIR) / "results" / "data_audit.csv", index=False)

print("modes:", audit["mode"].value_counts().to_dict())
print("broken images:", (~audit.image_ok).sum(), "| broken / mismatched trimaps:", (~audit.trimap_ok).sum())
print(audit[["fg_frac", "bg_frac", "border_frac"]].describe().round(3))
problems = audit[~audit.image_ok | ~audit.trimap_ok | (audit.fg_frac < 0.02)]
problems
'''),
    code('''
import matplotlib.pyplot as plt
import numpy as np
from bgbias import compose

def show_overlays(ids, cols=6, title=None):
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, image_id in zip(axes.flat, ids):
        img = np.asarray(compose.load_image(PET_ROOT / "images" / f"{image_id}.jpg"), dtype=float) / 255
        tri = compose.load_trimap(PET_ROOT / "annotations" / "trimaps" / f"{image_id}.png")
        overlay = img.copy()
        overlay[tri == 2] *= 0.35                         # background dimmed
        overlay[tri == 3] = [1.0, 0.1, 0.1]               # border in red
        ax.imshow(overlay); ax.set_title(image_id, fontsize=8)
    for ax in axes.flat: ax.axis("off")
    if title: fig.suptitle(title)
    plt.tight_layout(); plt.show()

rng = np.random.default_rng(0)
show_overlays(rng.choice(all_ids, 24, replace=False), title="Random masks: background dimmed, border red")
'''),
    code('''
# Smallest pet regions — most likely to break after background replacement.
show_overlays(audit.nsmallest(12, "fg_frac").image_id.tolist(), title="Smallest foreground fraction")
'''),
    md('''
### Граница trimap: включать в животное или нет

Слева — только класс 1, справа — класс 1 + граница (3). Фон ярко-зелёный, чтобы были видны края.
Решение записываем в `PLAN.md` (по умолчанию `include_border=True`, `blur_sigma=1.5`).
'''),
    code('''
green = (40, 200, 60)
ids = rng.choice(all_ids, 6, replace=False)
fig, axes = plt.subplots(len(ids), 3, figsize=(10, 3.2 * len(ids)))
for row, image_id in zip(axes, ids):
    img = compose.load_image(PET_ROOT / "images" / f"{image_id}.jpg")
    tri = compose.load_trimap(PET_ROOT / "annotations" / "trimaps" / f"{image_id}.png")
    row[0].imshow(img); row[0].set_title(image_id, fontsize=8)
    for ax, border in zip(row[1:], (False, True)):
        alpha = compose.foreground_alpha(tri, include_border=border, blur_sigma=1.5)
        ax.imshow(compose.foreground_only(img, alpha, color=green))
        ax.set_title(f"include_border={border}", fontsize=8)
for ax in axes.flat: ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    md("## 3. Пулы фонов (Places365)"),
    code('''
from bgbias import backgrounds as B

index = B.load_places_index(PLACES_ROOT)
pools = B.make_background_pools(index, fractions=(0.6, 0.15, 0.25), seed=SEED)
pools.to_csv(Path(REPO_DIR) / "manifests" / "background_pools.csv", index=False)

excluded = sorted(set(index.category) - set(pools.category))
print(f"{index.category.nunique()} categories, excluded {len(excluded)}:", excluded)
pools.groupby("pool").agg(images=("bg_path", "size"), categories=("category", "nunique"))
'''),
    code('''
fig, axes = plt.subplots(3, 8, figsize=(20, 8))
for row, pool in zip(axes, B.POOLS):
    sample = pools[pools.pool == pool].sample(8, random_state=0)
    for ax, (_, r) in zip(row, sample.iterrows()):
        ax.imshow(compose.load_image(PLACES_ROOT / r.bg_path))
        ax.set_title(f"{pool}: {r.category}", fontsize=7); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    md("## 4. Фиксированные фоны для val и test"),
    code('''
manifests = {
    "val": B.make_eval_manifest(splits["val"], pools, "val", seed=SEED),
    "test": B.make_eval_manifest(splits["test"], pools, "test", seed=SEED + 1),
}
for name, m in manifests.items():
    m.to_csv(Path(REPO_DIR) / "manifests" / f"{name}_swapped.csv", index=False)
    print(f"{name}: {len(m)} images, {m.bg_path.nunique()} unique backgrounds, {m.bg_category.nunique()} categories")
manifests["test"].head()
'''),
    md("## 5. Галерея примеров"),
    code('''
from bgbias.datasets import PetBackgroundDataset

def make_gallery(df, manifest, n=24, seed=0, path=None):
    sample = df.sample(n, random_state=seed).reset_index(drop=True)
    views = {c: PetBackgroundDataset(sample, PET_ROOT, condition=c, places_root=PLACES_ROOT, manifest=manifest)
             for c in ("original", "swapped", "fgonly", "bgonly")}
    cols = ["original", "trimap", "swapped", "fgonly", "bgonly"]
    fig, axes = plt.subplots(n, len(cols), figsize=(3 * len(cols), 2.8 * n))
    for i in range(n):
        image_id = sample.at[i, "image_id"]
        tri = compose.load_trimap(PET_ROOT / "annotations" / "trimaps" / f"{image_id}.png")
        for j, col in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(tri, cmap="viridis") if col == "trimap" else ax.imshow(views[col].render(i))
            ax.axis("off")
            if i == 0: ax.set_title(col, fontsize=12)
        if i > 0: axes[i, 0].set_title(image_id, fontsize=8)
    plt.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.show()

gallery_path = Path(REPO_DIR) / "report" / "figures" / "week1_gallery.jpg"
make_gallery(splits["test"], manifests["test"], n=24, path=gallery_path)
print("saved", gallery_path)
'''),
    md("## 6. Smoke test: DataLoader с заменой фона p = 0.5"),
    code('''
import time, torch
from torch.utils.data import DataLoader
from bgbias.datasets import build_transform, IMAGENET_MEAN, IMAGENET_STD
from bgbias.utils import seed_everything

seed_everything(0)
train_bg = pools.loc[pools.pool == "train", "bg_path"].tolist()
train_ds = PetBackgroundDataset(splits["train"], PET_ROOT, condition="original", transform=build_transform(train=True),
                                places_root=PLACES_ROOT, bg_pool=train_bg, p_swap=0.5)
loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2,
                    generator=torch.Generator().manual_seed(0))

t0 = time.time()
for i, (x, y) in enumerate(loader):
    if i == 4: break
print(f"batch {tuple(x.shape)}, labels {tuple(y.shape)}, {(time.time() - t0) / 5:.2f} s/batch")

breed_of = dict(zip(splits["train"].label, splits["train"].breed))
mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
for ax, img, label in zip(axes.flat, x[:16], y[:16]):
    ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0))
    ax.set_title(breed_of[label.item()], fontsize=8)
    ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    md('''
## 7. Сохранение

Все файлы уже лежат в `/content/dl_project`. Копия идёт в `MyDrive/bgbias/week1/`.

Чтобы отправить их в GitHub (ветка `data/week1`): добавь GitHub token с правом `repo` в **Colab Secrets** (значок ключа слева) под именем `GITHUB_TOKEN`, поставь `PUSH = True` и запусти ячейку.
'''),
    code('''
import shutil
OUTPUTS = ["splits", "manifests", "results/data_audit.csv", "report/figures/week1_gallery.jpg"]
for rel in OUTPUTS:
    src, dst = Path(REPO_DIR) / rel, OUT_DRIVE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True) if src.is_dir() else shutil.copy(src, dst)
print("copied to", OUT_DRIVE)

PUSH = False
GIT_NAME, GIT_EMAIL = "Colab", "colab@users.noreply.github.com"   # put your own
if PUSH:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
    def git(*args):
        r = subprocess.run(["git", "-C", REPO_DIR, *args], capture_output=True, text=True)
        print((r.stdout + r.stderr).replace(token, "***"))
        r.check_returncode()
    git("checkout", "-B", "data/week1")
    git("add", *OUTPUTS)
    git("-c", f"user.name={GIT_NAME}", "-c", f"user.email={GIT_EMAIL}",
        "commit", "-m", "Week 1: splits, background pools, eval manifests, data audit, gallery")
    git("push", f"https://{token}@github.com/mundmina/dl_project.git", "HEAD:data/week1")
'''),
])

(OUT / "00_setup.ipynb").write_text(json.dumps(setup_nb, indent=1, ensure_ascii=False))
(OUT / "01_data.ipynb").write_text(json.dumps(data_nb, indent=1, ensure_ascii=False))
print("ok")


PROJECT = code('''
from pathlib import Path
from bgbias.project import load_project_data
from bgbias.utils import DRIVE_ROOT

data = load_project_data(REPO_DIR)   # splits / manifests from the repo or from MyDrive/bgbias/week1
RUNS = DRIVE_ROOT / "runs"           # checkpoints and per-run outputs live on Drive
RUNS.mkdir(parents=True, exist_ok=True)
from bgbias.train import prune_last_checkpoints
# last.pt (weights + optimizer state) is only needed to resume; finished runs keep best.pt only
print(f"Drive space freed: {prune_last_checkpoints(RUNS) / 1e9:.2f} GB")
{name: len(df) for name, df in data.splits.items()}
''')

hog_nb = nb([
    md(f'''
# 02 · HOG + linear SVM baseline
{colab_badge("02_hog_svm.ipynb")}

Классический baseline. Обучение только на оригинальных train-изображениях, `C` выбирается по validation.
Тест здесь **не** запускается — финальная оценка всех моделей в `04_evaluate`.

GPU не нужен (можно CPU runtime). Модель сохраняется в `MyDrive/bgbias/runs/hog_svm/`.
'''),
    SETUP,
    PROJECT,
    code('''
import json, time
import joblib
import numpy as np
import pandas as pd
from bgbias import hog_svm
from bgbias.project import make_dataset

RUN_DIR = RUNS / "hog_svm"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CACHE = Path("/content/cache/hog")   # features are large, keep them on local disk only
CACHE.mkdir(parents=True, exist_ok=True)

def features(split, condition):
    path = CACHE / f"{split}_{condition}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    X = hog_svm.extract(make_dataset(data, split, condition=condition, transform=None))
    np.save(path, X)
    print(f"{split}/{condition}: {X.shape} in {time.time() - t0:.0f}s")
    return X

X_train, y_train = features("train", "original"), data.splits["train"].label.to_numpy()
X_val, y_val = features("val", "original"), data.splits["val"].label.to_numpy()
'''),
    md("## Подбор C по validation"),
    code('''
rows = []
for C in hog_svm.C_GRID:
    t0 = time.time()
    clf = hog_svm.make_svm(C).fit(X_train, y_train)
    rows.append({"C": C, "train_acc": clf.score(X_train, y_train), "val_acc": clf.score(X_val, y_val),
                 "fit_s": round(time.time() - t0)})
    print(rows[-1])
grid = pd.DataFrame(rows)
best_C = float(grid.loc[grid.val_acc.idxmax(), "C"])
grid.to_csv(RUN_DIR / "c_grid.csv", index=False)
print("best C:", best_C)
grid
'''),
    md("## Финальная модель + проверка на val swapped (только для информации)"),
    code('''
clf = hog_svm.make_svm(best_C).fit(X_train, y_train)
X_val_swapped = features("val", "swapped")
val_metrics = {
    "C": best_C,
    "val_acc_original": clf.score(X_val, y_val),
    "val_acc_swapped": clf.score(X_val_swapped, y_val),
}
joblib.dump(clf, RUN_DIR / "model.joblib")
(RUN_DIR / "val_metrics.json").write_text(json.dumps(val_metrics, indent=1))
val_metrics
'''),
])

def train_cells(configs, seeds_note=""):
    """Shared body of the training notebooks: train every config x seed, summary, push."""
    return [
        code(f'''
CONFIGS = {configs!r}
SEEDS = [0, 1, 2]
'''),
        SETUP,
        code('''
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
'''),
        PROJECT,
        code('''
import pandas as pd
from bgbias.train import load_config, train

configs = {name: load_config(f"{REPO_DIR}/configs/{name}.yaml") for name in CONFIGS}
pd.DataFrame(configs)   # paired configs (X_A / X_B) must differ only in name and p_swap
'''),
        code('''
# seed-major order: if Colab stops early, every config already has seed 0
histories = {}
for seed in SEEDS:
    for name, cfg in configs.items():
        run_dir = RUNS / f"{name}_seed{seed}"
        print(f"=== {run_dir.name} ===")
        histories[(name, seed)] = pd.DataFrame(train(cfg, seed, data, run_dir))
'''),
        code('''
rows = []
for (name, seed), h in histories.items():
    best = h.loc[h.val_acc_original.idxmax()]
    rows.append({"model": name, "seed": seed, "best_epoch": int(best.epoch),
                 "val_original": best.val_acc_original, "val_swapped": best.val_acc_swapped,
                 "gap": best.val_acc_original - best.val_acc_swapped})
best = pd.DataFrame(rows)
print(best.groupby("model")[["val_original", "val_swapped", "gap"]].agg(["mean", "std"]).round(4))
best.round(4)
'''),
        code('''
from bgbias.github_sync import sync
sync(REPO_DIR, "Training histories: " + ", ".join(CONFIGS))
'''),
    ]


RESUME_NOTE = """
Уже обученные запуски пропускаются, поэтому повторный Run all безопасен.
Если Colab отключился — просто запусти ноутбук снова, обучение продолжится с последней эпохи.
Checkpoints: `MyDrive/bgbias/runs/<name>_seed<k>/`. В конце истории обучения пушатся в GitHub.
После всех 03* ноутбуков — `04_evaluate`, затем `05_analysis`.
"""

train_nb = nb([
    md(f'''
# 03 · Train ResNet-18: A, B and the p ablation
{colab_badge("03_train_resnet.ipynb")}

**Runtime → T4 GPU → Run all.** Обучает все конфигурации, seeds 0, 1, 2 (~8–11 мин на запуск на T4).
Уже обученные запуски пропускаются, поэтому повторный Run all безопасен.

- `resnet_A` (p = 0) и `resnet_B` (p = 0.5) — основной эксперимент;
- `resnet_p025` (p = 0.25) и `resnet_p100` (p = 1.0) — ablation по вероятности замены фона.

В конце история обучения автоматически пушится в GitHub (нужен Colab secret `GITHUB_TOKEN`).

Конфиг зафиксирован после пробного запуска A/seed 0 (val original 92.75%, плато с ~3-й эпохи, 15 эпох достаточно).
Менять `epochs` / `lr` можно только в **обоих** `configs/resnet_A.yaml` и `configs/resnet_B.yaml` одновременно.

Checkpoints: `MyDrive/bgbias/runs/<name>_seed<k>/` (`best.pt` по val original, `last.pt` для продолжения; после
окончания прогона `last.pt` удаляется, остаётся только `best.pt`).
Если Colab отключился — просто запусти ноутбук снова, обучение продолжится с последней эпохи.
'''),
    *train_cells(["resnet_A", "resnet_B", "resnet_p025", "resnet_p100"]),
])

methods_nb = nb([
    md(f'''
# 03b · Train ResNet-18: other ways to fight background bias
{colab_badge("03b_train_methods.ipynb")}

**Runtime → T4 GPU → Run all.** 3 конфигурации × 3 seed, ~1.5–2 ч. Всё как у `resnet_B`, меняется только то,
что делаем с фоном во время обучения:

- `resnet_gray` — с вероятностью 0.5 фон заменяется серым (просто убрать фон);
- `resnet_blur` — с вероятностью 0.5 собственный фон размывается (животное остаётся чётким);
- `resnet_cutmix` — фон не трогаем, обычная аугментация CutMix (p = 0.5 на батч): помогает ли
  общая аугментация так же, как целевая?

Учти при анализе: `resnet_gray` видит при обучении ровно тот же серый фон, что и тестовое условие fg-only.
{RESUME_NOTE}
'''),
    *train_cells(["resnet_gray", "resnet_blur", "resnet_cutmix"]),
])

arch_nb = nb([
    md(f'''
# 03c · Train ConvNeXt-Tiny and Swin-T (A and B)
{colab_badge("03c_train_convnext_swin.ipynb")}

**Runtime → T4 GPU → Run all.** 4 конфигурации × 3 seed, ~3–4 ч (можно в несколько заходов).
Тот же протокол, что у ResNet-18 (ImageNet-веса, 15 эпох, AdamW, тот же lr); A и B отличаются только `p_swap`.
Вопрос: зависит ли смещение на фон от архитектуры — современная CNN (ConvNeXt) vs трансформер (Swin)?
{RESUME_NOTE}
'''),
    *train_cells(["convnext_A", "convnext_B", "swin_A", "swin_B"]),
])

depth_nb = nb([
    md(f'''
# 03d · Train ResNet-50 and ResNet-18 from scratch (A and B)
{colab_badge("03d_train_resnet50_scratch.ipynb")}

**Runtime → T4 GPU → Run all.** 4 конфигурации × 3 seed, ~3.5–4.5 ч (можно в несколько заходов).

- `resnet50_A/B` — та же семья, модель больше: влияет ли размер?
- `scratch_A/B` — ResNet-18 без ImageNet-предобучения (40 эпох, lr 1e-3 для всех слоёв). Точность будет
  заметно ниже (~30–50%); смотрим не на неё, а на то, сильнее ли такая модель опирается на фон.
{RESUME_NOTE}
'''),
    *train_cells(["resnet50_A", "resnet50_B", "scratch_A", "scratch_B"]),
])

eval_nb = nb([
    md(f'''
# 04 · Evaluate all models
{colab_badge("04_evaluate.ipynb")}

Оценивает каждый `runs/*_seed*/best.pt` и `runs/hog_svm/model.joblib` в условиях
original / swapped / fgonly / bgonly и собирает главную таблицу (mean ± std по seed).
Уже посчитанные предсказания берутся из `MyDrive/bgbias/results/<split>/`, новые модели (ablation) досчитываются.
В конце результаты автоматически пушатся в GitHub.

**`SPLIT = "test"`** — финальная оценка. Валидационный прогон сделан 2026-10-22 (`results/val/`), конфиги
зафиксированы, все seed обучены. После test ничего не подбираем и не переобучаем.
'''),
    code('''
SPLIT = "test"   # "val" for development runs
'''),
    SETUP,
    PROJECT,
    code('''
import joblib
import pandas as pd
import torch
from bgbias import evaluate as E, hog_svm
from bgbias.models import load_checkpoint
from bgbias.project import make_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(REPO_DIR) / "results" / SPLIT
OUT.mkdir(parents=True, exist_ok=True)
summaries = {}

CACHE = DRIVE_ROOT / "results" / SPLIT   # predictions already computed in an earlier run are reused

for run in sorted(p.parent for p in RUNS.glob("*_seed*/best.pt")):
    cached = CACHE / f"{run.name}_predictions.csv"
    if cached.exists():
        table = pd.read_csv(cached)
    else:
        model, cfg = load_checkpoint(run / "best.pt", data.num_classes, device)
        predict = E.torch_predict_fn(model, data, SPLIT, device, include_border=cfg["include_border"],
                                     blur_sigma=cfg["blur_sigma"], img_size=cfg["img_size"])
        table = E.predictions_table(data, SPLIT, predict)
    table.to_csv(OUT / f"{run.name}_predictions.csv", index=False)
    summaries[run.name] = E.summarize(table)
    print(run.name, {k: round(v, 4) for k, v in summaries[run.name].items() if k.startswith(("acc", "gap"))})

hog_path = RUNS / "hog_svm" / "model.joblib"
if (CACHE / "hog_svm_predictions.csv").exists():
    table = pd.read_csv(CACHE / "hog_svm_predictions.csv")
elif hog_path.exists():
    clf = joblib.load(hog_path)
    predict = lambda c: clf.predict(hog_svm.extract(make_dataset(data, SPLIT, condition=c, transform=None)))
    table = E.predictions_table(data, SPLIT, predict)
if hog_path.exists():
    table.to_csv(OUT / "hog_svm_predictions.csv", index=False)
    summaries["hog_svm"] = E.summarize(table)
    print("hog_svm", {k: round(v, 4) for k, v in summaries["hog_svm"].items() if k.startswith(("acc", "gap"))})
'''),
    md("## Главная таблица"),
    code('''
per_run = pd.DataFrame(summaries).T
per_run.to_csv(OUT / "summary_per_run.csv")

COLS = ["acc_original", "acc_swapped", "acc_fgonly", "acc_bgonly", "f1_original", "f1_swapped",
        "gap_swap", "gap_fg", "flip_swap"]
model = per_run.index.str.replace(r"_seed\\d+$", "", regex=True)
mean = per_run.groupby(model)[COLS].mean()
std = per_run.groupby(model)[COLS].std().fillna(0)
main = mean.map(lambda v: f"{v:.3f}") + " ± " + std.map(lambda v: f"{v:.3f}")
main["seeds"] = per_run.groupby(model).size()
main.to_csv(OUT / "main_table.csv")
main
'''),
    code('''
import shutil
from bgbias.github_sync import sync

dst = DRIVE_ROOT / "results" / SPLIT
shutil.copytree(OUT, dst, dirs_exist_ok=True)
print("saved to", OUT, "and", dst)
sync(REPO_DIR, f"Evaluation results ({SPLIT})")
'''),
])

(OUT / "02_hog_svm.ipynb").write_text(json.dumps(hog_nb, indent=1, ensure_ascii=False))
(OUT / "03_train_resnet.ipynb").write_text(json.dumps(train_nb, indent=1, ensure_ascii=False))
(OUT / "03b_train_methods.ipynb").write_text(json.dumps(methods_nb, indent=1, ensure_ascii=False))
(OUT / "03c_train_convnext_swin.ipynb").write_text(json.dumps(arch_nb, indent=1, ensure_ascii=False))
(OUT / "03d_train_resnet50_scratch.ipynb").write_text(json.dumps(depth_nb, indent=1, ensure_ascii=False))
(OUT / "04_evaluate.ipynb").write_text(json.dumps(eval_nb, indent=1, ensure_ascii=False))
print("ok 02-04")


analysis_nb = nb([
    md(f'''
# 05 · Analysis: figures, statistics, Grad-CAM
{colab_badge("05_analysis.ipynb")}

**Runtime → T4 GPU → Run all.** Использует сохранённые test-предсказания из `04_evaluate`
(`MyDrive/bgbias/results/test/`) и checkpoints из `MyDrive/bgbias/runs/`. Ничего не обучает и не подбирает.

Результат: рисунки в `report/figures/`, таблицы в `results/analysis/` — всё автоматически пушится в GitHub.

1. Точность по условиям (A vs B)
2. Ablation по p
3. Bootstrap-интервал для разницы gap(A) − gap(B)
4. Какие породы сильнее зависят от фона; кошки vs собаки
5. Примеры: A ошибается после замены фона, B — нет; Grad-CAM
6. Grad-CAM в цифрах: доля внимания на животном
7. Архитектуры: ResNet-18 / scratch / ResNet-50 / ConvNeXt-T / Swin-T, A vs B
8. Методы: CutMix, размытый фон, серый фон, saliency penalty, замена фона
'''),
    SETUP,
    code('''
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
'''),
    PROJECT,
    code('''
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from bgbias import analysis as A, evaluate as E
from bgbias.gradcam import GradCAM, eval_mask_transform, foreground_share
from bgbias.models import load_checkpoint
from bgbias.project import make_dataset

SPLIT = "test"
PRED_DIR = DRIVE_ROOT / "results" / SPLIT
FIG_DIR = Path(REPO_DIR) / "report" / "figures"
TAB_DIR = Path(REPO_DIR) / "results" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda")

from bgbias.plotting import COLORS, INK, INK2, LABELS, RULE, SURFACE, saver, use_style

use_style()
save = saver(FIG_DIR)

tables = {p.name.removesuffix("_predictions.csv"): pd.read_csv(p) for p in sorted(PRED_DIR.glob("*_predictions.csv"))}
per_run = pd.DataFrame({run: E.summarize(t) for run, t in tables.items()}).T
per_run["model"] = [A.model_name(r) for r in per_run.index]
per_run.to_csv(TAB_DIR / "summary_per_run.csv")
print(sorted(tables))
'''),
    md("## 1. Точность по условиям"),
    code('''
CONDS = [("original", "Original"), ("swapped", "Background swapped"), ("fgonly", "Foreground only")]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for model, offset in (("resnet_A", -0.12), ("resnet_B", 0.12)):
    runs = per_run[per_run.model == model]
    mean = np.array([runs[f"acc_{c}"].mean() for c, _ in CONDS])
    std = np.array([runs[f"acc_{c}"].std() for c, _ in CONDS])
    x = np.arange(len(CONDS)) + offset
    ax.errorbar(x, mean, yerr=std, fmt="o", ms=8, color=COLORS[model], ecolor=COLORS[model],
                elinewidth=1.5, capsize=3, label=LABELS[model])
    for xi, v in zip(x, mean):
        ax.annotate(f"{v:.1%}", (xi, v), xytext=(10 if offset > 0 else -10, 0), textcoords="offset points",
                    ha="left" if offset > 0 else "right", va="center", fontsize=8, color=INK2)
ax.set_xticks(range(len(CONDS)), [label for _, label in CONDS])
ax.set_xlim(-0.6, len(CONDS) - 0.4)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax.set_ylabel("Test accuracy (mean ± std, 3 seeds)")
ax.set_title("Background swap hurts model A; model B is almost unaffected")
ax.legend(loc="lower left")
save(fig, "fig1_accuracy_by_condition")
'''),
    md("## 2. Ablation: вероятность замены фона p"),
    code('''
abl = per_run[per_run.model.isin(A.P_SWAP)].copy()
abl["p"] = abl.model.map(A.P_SWAP)
g = abl.groupby("p")[["acc_original", "acc_swapped", "gap_swap"]].agg(["mean", "std"])
g.to_csv(TAB_DIR / "ablation_p.csv")
print("p values available:", list(g.index))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
panels = [("acc_original", "Accuracy on original images"), ("gap_swap", "Robustness gap: original − swapped")]
for ax, (metric, title) in zip(axes, panels):
    ax.errorbar(g.index, g[(metric, "mean")], yerr=g[(metric, "std")], fmt="-o", ms=8, color=INK2, capsize=3)
    for p, v in zip(g.index, g[(metric, "mean")]):
        tag = {0.0: "A", 0.5: "B"}.get(p)
        if tag:
            ax.annotate(tag, (p, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(g.index)
    ax.set_xlabel("Probability of background swap during training")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_title(title)
axes[1].set_ylim(bottom=min(0, axes[1].get_ylim()[0]))
fig.tight_layout()
save(fig, "fig2_ablation_p")
g.round(4)
'''),
    md("## 3. Статистика: bootstrap по тестовым изображениям"),
    code('''
stats = A.bootstrap_gap_difference(tables, "resnet_A", "resnet_B", n_boot=2000)
(TAB_DIR / "bootstrap_gap_difference.json").write_text(json.dumps(stats, indent=1))
print(f"gap A = {stats['gap_A']:.2%}, gap B = {stats['gap_B']:.2%}")
print(f"gap A - gap B = {stats['difference']:.2%}, 95% CI [{stats['ci_low']:.2%}, {stats['ci_high']:.2%}], "
      f"share of bootstrap samples <= 0: {stats['p_le_0']:.4f}")
'''),
    md("## 4. Какие породы зависят от фона сильнее всего"),
    code('''
gap_a, gap_b = A.per_breed_gap(tables, "resnet_A"), A.per_breed_gap(tables, "resnet_B")
breeds = pd.DataFrame({"gap_A": gap_a, "gap_B": gap_b}).sort_values("gap_A")
species = tables[next(iter(tables))].groupby("breed").species.first()
breeds["species"] = species
breeds.to_csv(TAB_DIR / "per_breed_gap.csv")

fig, ax = plt.subplots(figsize=(7.5, 10))
y = np.arange(len(breeds))
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True, color="#ebeae6")
ax.hlines(y, breeds.gap_B, breeds.gap_A, color=RULE, lw=2, zorder=1)
for model, col in (("resnet_A", "gap_A"), ("resnet_B", "gap_B")):
    ax.plot(breeds[col], y, "o", ms=7, color=COLORS[model], label=LABELS[model], zorder=2,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(y, [f"{b.replace('_', ' ')} ({s})" for b, s in zip(breeds.index, breeds.species)], fontsize=8)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Accuracy drop when the background is swapped (mean of 3 seeds)")
ax.set_title("Per-breed background dependence")
ax.legend(loc="upper left")   # top-left is empty: the most affected breeds sit at the top right
save(fig, "fig3_per_breed_gap")

print(breeds.groupby("species")[["gap_A", "gap_B"]].mean().round(4))
breeds.sort_values("gap_A", ascending=False).head(8).round(3)
'''),
    md("## 5. Примеры: A ошибается после замены фона, B — нет (Grad-CAM, seed 0)"),
    code('''
idx, n_total = A.flip_examples(tables, "resnet_A", "resnet_B", n=6, seed=0)
print(f"{n_total} test images: every A seed is right on original and wrong on swapped, every B seed right on both")

breed_of = dict(zip(data.splits[SPLIT].label, data.splits[SPLIT].breed))
models = {m: load_checkpoint(RUNS / f"{m}_seed0" / "best.pt", data.num_classes, device)[0]
          for m in ("resnet_A", "resnet_B")}
views = {c: make_dataset(data, SPLIT, condition=c) for c in ("original", "swapped")}
geometry = eval_mask_transform()

def cam_overlay(img, cam):
    """Unattended regions are darkened; attended regions keep full brightness."""
    arr = np.asarray(geometry(img), dtype=np.float32) / 255
    return (arr * (0.15 + 0.85 * cam[..., None])).clip(0, 1)

cols = [("original", None), ("original", "resnet_A"), ("original", "resnet_B"),
        ("swapped", None), ("swapped", "resnet_A"), ("swapped", "resnet_B")]
fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.3 * len(cols), 2.5 * len(idx)), squeeze=False)
for c in ("original", "swapped"):
    x = torch.stack([views[c][i][0] for i in idx]).to(device)
    for m, model in models.items():
        cam = GradCAM(model)
        maps, preds = cam(x)
        cam.remove()
        for r, i in enumerate(idx):
            j = cols.index((c, m))
            ax = axes[r, j]
            ax.imshow(cam_overlay(views[c].render(i), maps[r].cpu().numpy()))
            ok = preds[r].item() == data.splits[SPLIT].label.iat[i]
            ax.set_title(f"{'A' if m == 'resnet_A' else 'B'}: {breed_of[preds[r].item()].replace('_', ' ')}"
                         f" {'✓' if ok else '✗'}", fontsize=8, fontweight="normal",
                         color=INK if ok else "#d03b3b", loc="center")
for r, i in enumerate(idx):
    for c in ("original", "swapped"):
        ax = axes[r, cols.index((c, None))]
        ax.imshow(geometry(views[c].render(i)))
        ax.set_title(f"{c}\\ntrue: {data.splits[SPLIT].breed.iat[i].replace('_', ' ')}",
                     fontsize=8, fontweight="normal", loc="center")
for ax in axes.flat:
    ax.axis("off")
fig.tight_layout()
save(fig, "fig4_flip_examples_gradcam")
'''),
    md('''
## 6. Grad-CAM в цифрах

Доля «внимания» Grad-CAM, попадающая на животное (по trimap-маске), на 500 случайных тестовых изображениях,
все seed каждой модели ResNet-18 (A, B и методы из `03b`, `03e`). Для saliency penalty это прямая проверка:
обучение штрафовало внимание на фон — видно ли это по Grad-CAM? Для сравнения — доля площади кадра, которую занимает животное.
Grad-CAM на рисунке выше — иллюстрация; вывод делаем по этим цифрам.
'''),
    code('''
rng = np.random.default_rng(0)
sample = np.sort(rng.choice(len(data.splits[SPLIT]), size=500, replace=False))
rows = []
CAM_MODELS = ["resnet_A", "resnet_B", "resnet_gray", "resnet_blur", "resnet_cutmix", "resnet_rrr1", "resnet_rrr10"]
for model_name in [m for m in CAM_MODELS if (RUNS / f"{m}_seed0" / "best.pt").exists()]:
    for seed in (0, 1, 2):
        if not (RUNS / f"{model_name}_seed{seed}" / "best.pt").exists():
            continue
        model, cfg = load_checkpoint(RUNS / f"{model_name}_seed{seed}" / "best.pt", data.num_classes, device)
        for c in ("original", "swapped"):
            share, area = foreground_share(model, data, SPLIT, c, sample, device)
            rows.append({"model": model_name, "seed": seed, "condition": c,
                         "cam_on_pet": share.mean(), "pet_area": area.mean()})
cam_stats = pd.DataFrame(rows)
cam_stats.to_csv(TAB_DIR / "gradcam_foreground_share.csv", index=False)
cam_stats.groupby(["model", "condition"])[["cam_on_pet", "pet_area"]].agg(["mean", "std"]).round(3)
'''),
    md('''
## 7. Архитектуры: A vs B для каждой

Для каждой архитектуры: модель без замены фона (A) и с заменой p = 0.5 (B), по 3 seed.
Показываются только уже обученные и оценённые модели (`03c`, `03d` → `04_evaluate`).
'''),
    code('''
def model_stats(model):
    runs = per_run[per_run.model == model]
    out = {"n_seeds": len(runs)}
    for m in ("acc_original", "acc_swapped", "acc_fgonly", "acc_bgonly", "gap_swap", "flip_swap"):
        out[m] = runs[m].mean()
        out[m + "_std"] = runs[m].std()
    return out

fam_rows = []
for fam, (ma, mb) in A.FAMILIES.items():
    if not (per_run.model == ma).any() or not (per_run.model == mb).any():
        print("skip", fam, "(not evaluated yet)")
        continue
    boot = A.bootstrap_gap_difference(tables, ma, mb, n_boot=2000)
    for variant, m in (("A", ma), ("B", mb)):
        fam_rows.append({"family": fam, "variant": variant, "model": m, **model_stats(m)})
    fam_rows[-1].update({"gapA_minus_gapB": boot["difference"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]})
families = pd.DataFrame(fam_rows)
families.to_csv(TAB_DIR / "architectures.csv", index=False)

fams = list(dict.fromkeys(families.family))
fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(fams) + 1.6), sharey=True)
y = np.arange(len(fams))
for ax, metric, title in ((axes[0], "acc_original", "Accuracy on original images"),
                          (axes[1], "gap_swap", "Robustness gap: original − swapped")):
    for variant, model_key, dy in (("A", "resnet_A", -0.12), ("B", "resnet_B", 0.12)):
        d = families[families.variant == variant].set_index("family").loc[fams]
        ax.errorbar(d[metric], y + dy, xerr=d[metric + "_std"], fmt="o", ms=7, color=COLORS[model_key],
                    capsize=3, label=f"{variant}: {'background swap p = 0.5' if variant == 'B' else 'original images'}")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0 if metric == "acc_original" else 1))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color="#ebeae6")
axes[1].axvline(0, color=INK2, lw=0.8)
axes[0].set_yticks(y, fams)
axes[0].invert_yaxis()
axes[1].legend(loc="lower right")
fig.tight_layout()
save(fig, "fig5_architectures")
families.drop(columns=[c for c in families.columns if c.endswith("_std")]).round(4)
'''),
    md('''
## 8. Методы против смещения на фон (ResNet-18)

Все модели — ResNet-18 с ImageNet-весами и одинаковым протоколом; отличается только обработка фона при обучении.
CutMix — общая аугментация, ничего не знает о фоне. `Gray background` видит при обучении тот же серый фон,
что и тестовое условие fg-only, поэтому его fg-only точность не сравнима с остальными.
'''),
    code('''
meth = pd.DataFrame([{"method": label, "model": m, **model_stats(m)}
                     for label, m in A.METHODS.items() if (per_run.model == m).any()])
meth.to_csv(TAB_DIR / "methods.csv", index=False)
for label, m in A.METHODS.items():
    if m != "resnet_A" and (per_run.model == m).any():
        b = A.bootstrap_gap_difference(tables, "resnet_A", m, n_boot=2000)
        print(f"{label:20s} gap A - gap method = {b['difference']:.2%}  95% CI [{b['ci_low']:.2%}, {b['ci_high']:.2%}]")

fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(meth) + 1.6), sharey=True)
y = np.arange(len(meth))
for ax, metric, title in ((axes[0], "acc_original", "Accuracy on original images"),
                          (axes[1], "gap_swap", "Robustness gap: original − swapped")):
    ax.errorbar(meth[metric], y, xerr=meth[metric + "_std"], fmt="o", ms=8, color=INK2, capsize=3)
    for yi, v in zip(y, meth[metric]):
        ax.annotate(f"{v:.1%}", (v, yi), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color="#ebeae6")
axes[1].axvline(0, color=INK2, lw=0.8)
axes[0].set_yticks(y, meth.method)
axes[0].invert_yaxis()
fig.tight_layout()
save(fig, "fig6_methods")
meth.drop(columns=[c for c in meth.columns if c.endswith("_std")]).round(4)
'''),
    md("## Сохранение и push"),
    code('''
import shutil
from bgbias.github_sync import sync

shutil.copytree(FIG_DIR, DRIVE_ROOT / "figures", dirs_exist_ok=True)
shutil.copytree(TAB_DIR, DRIVE_ROOT / "results" / "analysis", dirs_exist_ok=True)
sync(REPO_DIR, "Analysis: figures, bootstrap, per-breed gaps, Grad-CAM")
'''),
])

(OUT / "05_analysis.ipynb").write_text(json.dumps(analysis_nb, indent=1, ensure_ascii=False))
print("ok 05")


rrr_nb = nb([
    md(f"""
# 03e · Train ResNet-18 with a saliency penalty ("right for the right reasons")
{colab_badge("03e_train_rrr.ipynb")}

**Runtime → T4 GPU → Run all.** 2 конфигурации × 3 seed, ~2.5–3.5 ч (можно в несколько заходов).

Другой подход: не менять данные, а менять функцию потерь (Ross et al., 2017). Модель видит только оригинальные фото,
как A, но к cross-entropy добавляется штраф: доля градиента по входу (saliency), попавшая на фон по маске.
Штраф лежит в [0, 1], поэтому λ сравнима с loss: `resnet_rrr1` (λ = 1) и `resnet_rrr10` (λ = 10).
Обучение в fp32 (двойной backward), поэтому медленнее остальных ResNet-18.
В истории обучения есть колонка `train_bg_share` — она должна падать.
{RESUME_NOTE}
"""),
    *train_cells(["resnet_rrr1", "resnet_rrr10"]),
])

external_nb = nb([
    md(f"""
# 04b · External test: Stanford Dogs (real photos, no cut-and-paste)
{colab_badge("04b_external_stanford_dogs.ipynb")}

**Runtime → T4 GPU → Run all.** ~40–60 мин; первый запуск скачивает Stanford Dogs (~780 MB) и кладёт в Drive.

Все наши «новые фоны» до сих пор были искусственными (вырезали и вставили). Здесь — настоящие фото 21 нашей
породы собак из другого датасета: другие фотографы, другие места. Обучение не трогаем, только оценка.

- `full` — фото целиком; `crop` — только рамка с собакой (большая часть фона убрана).
- `acc` — по всем 37 классам; `acc_dog` — только среди 21 породы, которые есть в Stanford Dogs.

Оговорка для отчёта: Stanford Dogs собран из ImageNet, поэтому ImageNet-предобученные модели видели эти фото
при предобучении. Это одинаково для A и B, так что сравнение честное; модели `scratch_*` их не видели.
"""),
    SETUP,
    code("""
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
"""),
    PROJECT,
    code("""
import numpy as np
import pandas as pd
from bgbias import analysis as A, stanford_dogs as SD
from bgbias.datasets import build_transform
from bgbias.models import load_checkpoint

SD_ROOT = SD.prepare_stanford_dogs()
breed_to_label = dict(zip(data.splits["test"].breed, data.splits["test"].label))
index = SD.load_index(SD_ROOT, breed_to_label)
print(len(index), "photos of", index.breed.nunique(), "breeds")
index.breed.value_counts().to_frame().T
"""),
    code("""
device = torch.device("cuda")
OUT = Path(REPO_DIR) / "results" / "stanford_dogs"
CACHE = DRIVE_ROOT / "results" / "stanford_dogs"   # predictions from an earlier session are reused
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
views = {v: SD.StanfordDogs(index, SD_ROOT, v, build_transform(train=False)) for v in ("full", "crop")}
dog_labels = sorted(index.label.unique())

tables, rows = {}, []
for run in sorted(p.parent for p in RUNS.glob("*_seed*/best.pt")):
    name = f"{run.name}_predictions.csv"
    if (CACHE / name).exists():
        table = pd.read_csv(CACHE / name)
    else:
        model, cfg = load_checkpoint(run / "best.pt", data.num_classes, device)
        table = index[["path", "breed", "label"]].copy()
        for v, ds in views.items():
            table[f"pred_{v}"], table[f"pred_dog_{v}"] = SD.predict(model, ds, device, dog_labels)
        table.to_csv(CACHE / name, index=False)
        del model
        torch.cuda.empty_cache()
    table.to_csv(OUT / name, index=False)
    tables[run.name] = table
    row = {"run": run.name, "model": A.model_name(run.name)}
    row.update({f"acc_{c}": (table[f"pred_{c}"] == table.label).mean() for c in ("full", "crop", "dog_full", "dog_crop")})
    rows.append(row)
    print(run.name, {k: round(v, 3) for k, v in row.items() if k.startswith("acc")})
per_run = pd.DataFrame(rows).set_index("run")
per_run.to_csv(OUT / "summary_per_run.csv")
"""),
    md("## A vs B на настоящих фото"),
    code("""
oxford = pd.read_csv(DRIVE_ROOT / "results" / "test" / "summary_per_run.csv", index_col=0)
oxford["model"] = [A.model_name(r) for r in oxford.index]

def mean_std(df, model, col):
    v = df.loc[df.model == model, col]
    return v.mean(), v.std()

rows = []
groups = {**{f: pair for f, pair in A.FAMILIES.items()},
          **{label: ("resnet_A", m) for label, m in A.METHODS.items() if m not in ("resnet_A", "resnet_B")}}
for label, (ma, mb) in groups.items():
    if not ((per_run.model == ma).any() and (per_run.model == mb).any()):
        continue
    boot = A.bootstrap_accuracy_difference(tables, ma, mb, "full")
    for variant, m in (("A", ma), ("B", mb)):
        r = {"group": label, "variant": variant, "model": m}
        for col in ("acc_full", "acc_crop", "acc_dog_full"):
            r[col], r[col + "_std"] = mean_std(per_run, m, col)
        r["oxford_acc_original"], _ = mean_std(oxford, m, "acc_original")
        rows.append(r)
    rows[-1].update({"B_minus_A_full": boot["difference"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]})
external = pd.DataFrame(rows)
TAB = Path(REPO_DIR) / "results" / "analysis"
TAB.mkdir(parents=True, exist_ok=True)
external.to_csv(TAB / "stanford_dogs.csv", index=False)
external.drop(columns=[c for c in external.columns if c.endswith("_std")]).round(3)
"""),
    code("""
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from bgbias.plotting import COLORS, INK2, saver, use_style

use_style()
FIG_DIR = Path(REPO_DIR) / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
save = saver(FIG_DIR)

fams = [f for f in A.FAMILIES if (external.group == f).any()]
d = external[external.group.isin(fams)]
fig, ax = plt.subplots(figsize=(8, 0.6 * len(fams) + 1.6))
y = np.arange(len(fams))
for variant, key, dy in (("A", "resnet_A", -0.12), ("B", "resnet_B", 0.12)):
    v = d[d.variant == variant].set_index("group").loc[fams]
    ax.errorbar(v.acc_full, y + dy, xerr=v.acc_full_std, fmt="o", ms=7, color=COLORS[key], capsize=3,
                label="A: original images" if variant == "A" else "B: background swap p = 0.5")
ax.set_yticks(y, fams)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.grid(axis="x", visible=True, color="#ebeae6")
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Accuracy on Stanford Dogs photos (37 classes, mean ± std over seeds)")
ax.set_title("External test on real photos: does background training transfer?")
ax.legend(loc="lower right")
save(fig, "fig8_stanford_dogs")

for label in groups:
    r = external[(external.group == label) & (external.variant == "B")]
    if len(r):
        r = r.iloc[0]
        print(f"{label:28s} B - A = {r.B_minus_A_full:+.2%}  95% CI [{r.ci_low:+.2%}, {r.ci_high:+.2%}]")
"""),
    code("""
import shutil
from bgbias.github_sync import sync

shutil.copytree(FIG_DIR, DRIVE_ROOT / "figures", dirs_exist_ok=True)
shutil.copy(TAB / "stanford_dogs.csv", DRIVE_ROOT / "results" / "analysis" / "stanford_dogs.csv")
sync(REPO_DIR, "External test on Stanford Dogs")
"""),
])

probing_nb = nb([
    md(f"""
# 05b · Probing: how much do the features know about the background?
{colab_badge("05b_probing.ipynb")}

**Runtime → T4 GPU → Run all.** ~40–60 мин. Ничего не обучает, кроме маленьких линейных классификаторов.

Берём признаки из предпоследнего слоя каждой модели на test-картинках с заменённым фоном и учим логистическую
регрессию угадывать **категорию сцены фона** (87 категорий Places365, случайное угадывание ≈ 1.1%), 5-fold CV.
Фон назначен случайно и с породой не связан, поэтому для распознавания пород эта информация бесполезна.
Чем лучше её можно достать из признаков, тем больше модель «держит в голове» фон.

Для сравнения — те же сети с ImageNet-весами без дообучения и случайно инициализированная ResNet-18.
"""),
    SETUP,
    code("""
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
"""),
    PROJECT,
    code("""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bgbias import analysis as A
from bgbias.models import build_model, load_checkpoint, strip_head
from bgbias.project import make_dataset
from bgbias.train import make_loader

device = torch.device("cuda")
SPLIT = "test"
scenes = data.manifests[SPLIT].set_index("image_id").loc[data.splits[SPLIT].image_id, "bg_category"].to_numpy()
print(len(set(scenes)), "scene categories, chance =", f"{1 / len(set(scenes)):.1%}")
loader = make_loader(make_dataset(data, SPLIT, condition="swapped"), 128, False, 0, 2)

@torch.no_grad()
def features(model):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()

def probe(feats):
    clf = make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegression(max_iter=2000, C=0.5))
    return cross_val_score(clf, feats, scenes, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
"""),
    code("""
CACHE = DRIVE_ROOT / "results" / "analysis" / "probing_runs.csv"   # finished runs are skipped on a rerun
CACHE.parent.mkdir(parents=True, exist_ok=True)
done = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["run", "model", "scene_acc"])

def record(run, model_name, make_model):
    global done
    if run in set(done.run):
        return
    model = make_model().to(device)
    acc = probe(features(model))
    done = pd.concat([done, pd.DataFrame([{"run": run, "model": model_name, "scene_acc": acc}])], ignore_index=True)
    done.to_csv(CACHE, index=False)
    print(f"{run:24s} scene accuracy {acc:.1%}")
    del model
    torch.cuda.empty_cache()

# references: ImageNet features without fine-tuning, and an untrained network
for arch in ("resnet18", "resnet50", "convnext_tiny", "swin_t"):
    record(f"imagenet_{arch}", f"imagenet_{arch}", lambda: strip_head(build_model(arch, data.num_classes), arch))
record("random_resnet18", "random_resnet18",
       lambda: strip_head(build_model("resnet18", data.num_classes, pretrained=False), "resnet18"))

for run in sorted(p.parent for p in RUNS.glob("*_seed*/best.pt")):
    def make(run=run):
        model, cfg = load_checkpoint(run / "best.pt", data.num_classes, device)
        return strip_head(model, cfg.get("arch", "resnet18"))
    record(run.name, A.model_name(run.name), make)
"""),
    code("""
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from bgbias.plotting import COLORS, INK2, saver, use_style

use_style()
FIG_DIR = Path(REPO_DIR) / "report" / "figures"
TAB = Path(REPO_DIR) / "results" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
save = saver(FIG_DIR)

stats = done.groupby("model").scene_acc.agg(["mean", "std", "count"])
stats["std"] = stats["std"].fillna(0)   # references have a single run
stats.to_csv(TAB / "probing.csv")
REF = {"ResNet-18": "imagenet_resnet18", "ResNet-18 (scratch)": "random_resnet18", "ResNet-50": "imagenet_resnet50",
       "ConvNeXt-T": "imagenet_convnext_tiny", "Swin-T": "imagenet_swin_t"}
fams = [f for f, (ma, mb) in A.FAMILIES.items() if ma in stats.index and mb in stats.index]
meths = [(label, m) for label, m in A.METHODS.items() if m in stats.index]

fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * max(len(fams), len(meths)) + 1.8))
ax = axes[0]
y = np.arange(len(fams))
for variant, key, dy in ((0, "resnet_A", -0.12), (1, "resnet_B", 0.12)):
    m = [A.FAMILIES[f][variant] for f in fams]
    ax.errorbar(stats.loc[m, "mean"], y + dy, xerr=stats.loc[m, "std"].fillna(0), fmt="o", ms=7,
                color=COLORS[key], capsize=3, label="A: original images" if variant == 0 else "B: background swap")
ref = [stats.loc[REF[f], "mean"] if REF[f] in stats.index else np.nan for f in fams]
ax.plot(ref, y, "|", ms=14, mew=2, color=INK2, label="before fine-tuning (ImageNet / random)")
ax.set_yticks(y, fams)
ax.invert_yaxis()
ax.set_title("Architectures")
ax.legend(loc="lower right", fontsize=8)
ax = axes[1]
y = np.arange(len(meths))
ax.errorbar([stats.loc[m, "mean"] for _, m in meths], y, xerr=[stats.loc[m, "std"] for _, m in meths],
            fmt="o", ms=7, color=INK2, capsize=3)
ax.set_yticks(y, [label for label, _ in meths])
ax.invert_yaxis()
ax.set_title("Methods (ResNet-18)")
for ax in axes:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color="#ebeae6")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Background scene decoded from features (linear probe, 87 classes)")
fig.tight_layout()
save(fig, "fig7_probing")
stats.round(3)
"""),
    code("""
import shutil
from bgbias.github_sync import sync

shutil.copytree(FIG_DIR, DRIVE_ROOT / "figures", dirs_exist_ok=True)
sync(REPO_DIR, "Probing: background scene decodability from features")
"""),
])

demo_nb = nb([
    md(f"""
# 06 · Demo: change the background, watch the prediction
{colab_badge("06_demo.ipynb")}

**Runtime → T4 GPU → Run all**, затем открой ссылку `https://….gradio.live` из последней ячейки
(работает 72 часа, можно открыть на защите с любого компьютера).

Загружаешь фото кошки или собаки → сеть сегментации (DeepLabV3) вырезает животное → выбираешь новый фон →
модели A и B (ResNet-18, seed 0) показывают топ-3 пород для оригинала и для нового фона.
"""),
    SETUP,
    code("""
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
"""),
    PROJECT,
    code("""
# Checkpoints from the teammate's archive (MyDrive/dl/runs.zip, a shortcut to the shared folder):
# copy only the two best.pt the demo needs into runs/, the 12.5 GB archive stays where it is.
import shutil, zipfile
ARCHIVE = Path("/content/drive/MyDrive/dl/runs.zip")
for m in ("resnet_A", "resnet_B"):
    dst = RUNS / f"{m}_seed0" / "best.pt"
    if dst.exists():
        continue
    assert ARCHIVE.exists(), f"no {dst} and no {ARCHIVE}: add the shared dl folder to My Drive"
    with zipfile.ZipFile(ARCHIVE) as z:
        member = next(n for n in z.namelist() if n.endswith(f"{m}_seed0/best.pt"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
    print("copied", member, "->", dst)
"""),
    code("""
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from bgbias import analysis as A, compose
from bgbias.datasets import build_transform
from bgbias.models import load_checkpoint

device = torch.device("cuda")
models = {m: load_checkpoint(RUNS / f"{m}_seed0" / "best.pt", data.num_classes, device)[0] for m in ("resnet_A", "resnet_B")}
breed_of = dict(zip(data.splits["test"].label, data.splits["test"].breed))
to_tensor = build_transform(train=False)

seg_weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
seg = deeplabv3_resnet50(weights=seg_weights).to(device).eval()
seg_tf = seg_weights.transforms()
PET_CLASSES = [seg_weights.meta["categories"].index(c) for c in ("cat", "dog")]

@torch.no_grad()
def pet_alpha(img):
    labels = seg(seg_tf(img)[None].to(device))["out"][0].argmax(0).cpu().numpy()
    mask = Image.fromarray(np.isin(labels, PET_CLASSES).astype(np.uint8) * 255).resize(img.size, Image.BILINEAR)
    return np.asarray(mask.filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float32) / 255

@torch.no_grad()
def top3(img):
    x = to_tensor(img)[None].to(device)
    out = {}
    for m, model in models.items():
        p = model(x).softmax(1)[0].cpu()
        out[m] = {breed_of[i].replace("_", " "): float(p[i]) for i in p.topk(3).indices.tolist()}
    return out

rng = np.random.default_rng(0)
bg_paths = [data.places_root / p for p in rng.choice(data.bg_pool("test"), 12, replace=False)]

# example photos: test images where every A seed flips on the swapped background and every B seed does not
tables = {p.name.removesuffix("_predictions.csv"): pd.read_csv(p)
          for p in (DRIVE_ROOT / "results" / "test").glob("resnet_[AB]_seed*_predictions.csv")}
idx, _ = A.flip_examples(tables, "resnet_A", "resnet_B", n=6, seed=1)
examples = [str(data.pet_root / "images" / f"{data.splits['test'].image_id.iat[i]}.jpg") for i in idx]
print(len(examples), "example photos")
"""),
    code("""
import gradio as gr

def run(pet, bg_index, bg_upload):
    if pet is None:
        raise gr.Error("Загрузи фото кошки или собаки")
    pet = pet.convert("RGB")
    alpha = pet_alpha(pet)
    if alpha.mean() < 0.01:
        raise gr.Error("Не нашёл на фото кошку или собаку")
    bg = bg_upload.convert("RGB") if bg_upload is not None else compose.load_image(bg_paths[bg_index or 0])
    swapped = compose.swap_background(pet, alpha, bg)
    before, after = top3(pet), top3(swapped)
    return swapped, before["resnet_A"], before["resnet_B"], after["resnet_A"], after["resnet_B"]

with gr.Blocks(title="Background bias demo") as demo:
    gr.Markdown("## Меняется ли порода, если поменять фон?\\n"
                "**A** обучена на обычных фото, **B** — с заменой фона в половине случаев.")
    bg_index = gr.State(0)
    with gr.Row():
        pet = gr.Image(type="pil", label="Фото питомца")
        with gr.Column():
            gallery = gr.Gallery(value=[str(p) for p in bg_paths], label="Новый фон (кликни)", columns=4,
                                 height=260, allow_preview=False)
            bg_upload = gr.Image(type="pil", label="…или загрузи свой фон", height=160)
    def pick(evt: gr.SelectData):   # gradio passes the click only to a parameter annotated SelectData
        return evt.index
    gallery.select(pick, None, bg_index)
    go = gr.Button("Поменять фон", variant="primary")
    swapped = gr.Image(label="С новым фоном", height=320)
    with gr.Row():
        a_before = gr.Label(num_top_classes=3, label="A — оригинал")
        b_before = gr.Label(num_top_classes=3, label="B — оригинал")
        a_after = gr.Label(num_top_classes=3, label="A — новый фон")
        b_after = gr.Label(num_top_classes=3, label="B — новый фон")
    gr.Examples(examples=[[e] for e in examples], inputs=pet, label="Примеры из test (A здесь ошибалась на чужом фоне)")
    go.click(run, [pet, bg_index, bg_upload], [swapped, a_before, b_before, a_after, b_after])

demo.launch(share=True)
"""),
])

(OUT / "03e_train_rrr.ipynb").write_text(json.dumps(rrr_nb, indent=1, ensure_ascii=False))
(OUT / "04b_external_stanford_dogs.ipynb").write_text(json.dumps(external_nb, indent=1, ensure_ascii=False))
(OUT / "05b_probing.ipynb").write_text(json.dumps(probing_nb, indent=1, ensure_ascii=False))
(OUT / "06_demo.ipynb").write_text(json.dumps(demo_nb, indent=1, ensure_ascii=False))
print("ok 03e, 04b, 05b, 06")


case_nb = nb([
    md(f"""
# 07 · Case study: Waterbirds — does our fix work on a standard shortcut benchmark?
{colab_badge("07_case_waterbirds.ipynb")}

**Runtime → T4 GPU → Run all.** ~1.5–2 ч: скачивание (~480 MB, один раз), маски (~10 мин, один раз), 2 модели × 3 seed.

Background bias на породах — маленький удобный случай большой проблемы shortcut learning. Здесь проверяем,
переносится ли наш метод на **Waterbirds** (Sagawa et al., 2020): птицы из CUB на фонах Places, и в train
95% водоплавающих птиц сидят на воде, а 95% сухопутных — на суше. Модель легко учится отличать воду от суши
вместо птиц. Главная метрика — **worst-group accuracy**: точность на самой трудной группе (обычно «водоплавающая
птица на суше»).

- `waterbirds_A` — обычное дообучение ResNet-18;
- `waterbirds_B` — наш метод без изменений: с вероятностью 0.5 фон заменяется случайной сценой Places365.
  Ручных масок нет — их делает сеть сегментации DeepLabV3 (класс «bird»).

Checkpoint выбирается по средней точности на val, без групповых меток (честное сравнение).
"""),
    SETUP,
    code("""
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
"""),
    PROJECT,
    code("""
import numpy as np
import pandas as pd
from bgbias import waterbirds as W
from bgbias.train import load_config

device = torch.device("cuda")
WB_ROOT = W.prepare_waterbirds()
wb = W.load_metadata(WB_ROOT)
print(pd.crosstab([wb.split, wb.y.map({0: "landbird", 1: "waterbird"})], wb.place.map({0: "land", 1: "water"})))
masks = W.compute_masks(WB_ROOT, wb, device, DRIVE_ROOT / "data" / "waterbirds_masks.npz")
found = np.mean([W.unpack_mask(m).mean() > 0.005 for m in masks.values()])
print(f"segmentation found the bird in {found:.1%} of images")
"""),
    code("""
import matplotlib.pyplot as plt
from bgbias.datasets import build_transform

bg_pool = data.bg_pool("train")
demo = W.WaterbirdsDataset(wb[wb.split == "train"].sample(6, random_state=0), WB_ROOT, lambda im: im, masks,
                           p_swap=1.0, bg_pool=bg_pool, places_root=data.places_root)
torch.manual_seed(0)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for i in range(6):
    axes[0, i].imshow(W.compose.load_image(WB_ROOT / demo.df.at[i, "img_filename"]))
    axes[0, i].set_title(W.GROUPS[demo.df.at[i, "group"]], fontsize=9)
    axes[1, i].imshow(demo.render(i))
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
    code("""
CONFIGS = ["waterbirds_A", "waterbirds_B"]
SEEDS = [0, 1, 2]
WB_RUNS = DRIVE_ROOT / "runs_waterbirds"   # kept apart: 04 / 04b / 05b scan runs/ for 37-class Pet models
configs = {name: load_config(f"{REPO_DIR}/configs/{name}.yaml") for name in CONFIGS}
for seed in SEEDS:
    for name, cfg in configs.items():
        run_dir = WB_RUNS / f"{name}_seed{seed}"
        print(f"=== {run_dir.name} ===")
        W.train(cfg, seed, wb, WB_ROOT, masks, data.places_root, bg_pool, run_dir, device)
"""),
    md("## Test: средняя точность и точность по группам"),
    code("""
from bgbias.models import build_model

test_loader = W._loader(W.WaterbirdsDataset(wb[wb.split == "test"], WB_ROOT, build_transform(train=False)), 128, False, 0)
rows = []
for name in CONFIGS:
    for seed in SEEDS:
        ckpt = torch.load(WB_RUNS / f"{name}_seed{seed}" / "best.pt", map_location=device, weights_only=False)
        model = build_model(ckpt["config"].get("arch", "resnet18"), 2, pretrained=False).to(device)
        model.load_state_dict(ckpt["model"])
        rows.append({"model": name, "seed": seed, **W.group_metrics(*W.predict(model, test_loader, device))})
res = pd.DataFrame(rows)
OUT = Path(REPO_DIR) / "results" / "waterbirds"
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "test_per_run.csv", index=False)
summary = res.drop(columns="seed").groupby("model").agg(["mean", "std"])
summary.to_csv(OUT / "test_summary.csv")
summary.round(3)
"""),
    code("""
from matplotlib.ticker import PercentFormatter
from bgbias.plotting import COLORS, saver, use_style

use_style()
FIG_DIR = Path(REPO_DIR) / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
save = saver(FIG_DIR)

cols = ["avg_acc", "worst_group_acc", *W.GROUPS.values()]
names = ["Average", "Worst group", *W.GROUPS.values()]
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(cols))
for name, key, dx in (("waterbirds_A", "resnet_A", -0.18), ("waterbirds_B", "resnet_B", 0.18)):
    m = res[res.model == name][cols]
    ax.bar(x + dx, m.mean(), width=0.34, yerr=m.std(), capsize=3, color=COLORS[key],
           label="A: plain fine-tuning" if name.endswith("A") else "B: random background swap (ours)")
    for xi, v in zip(x + dx, m.mean()):
        ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
ax.set_xticks(x, names, fontsize=9)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_ylim(0, 1.05)
ax.set_ylabel("Test accuracy (mean ± std, 3 seeds)")
ax.set_title("Waterbirds: background swap and the hardest groups")
ax.legend(loc="lower left")
save(fig, "fig9_waterbirds")
"""),
    code("""
import shutil
from bgbias.github_sync import sync

for name in CONFIGS:
    for seed in SEEDS:
        src = WB_RUNS / f"{name}_seed{seed}" / "history.json"
        dst = OUT / "runs" / f"{name}_seed{seed}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
shutil.copytree(OUT, DRIVE_ROOT / "results" / "waterbirds", dirs_exist_ok=True)
shutil.copytree(FIG_DIR, DRIVE_ROOT / "figures", dirs_exist_ok=True)
sync(REPO_DIR, "Case study: Waterbirds")
"""),
])

(OUT / "07_case_waterbirds.ipynb").write_text(json.dumps(case_nb, indent=1, ensure_ascii=False))
print("ok 07")



attribution_nb = nb([
    md(f'''
# 08 · Attribution: how much of the decision comes from the pet vs the background
{colab_badge("08_attribution.ipynb")}

**Runtime → T4 GPU → Run all.** Ничего не обучает. Checkpoints берутся из архива тиммейта
`MyDrive/dl/runs.zip` (только нужные `best.pt`, на локальный диск Colab, Drive не занимает).
Если какого-то run в архиве нет — берётся `MyDrive/bgbias/runs/<run>/best.pt`.

Единица измерения везде одна: **log p(верный класс | картинка)**.

1. **Region Shapley** (все 13 моделей × 3 seed, весь test): каждую картинку показываем в 4 вариантах —
   целиком, только животное, только фон, всё серое — и честно делим решение на вклад животного φ_pet
   и вклад фона φ_bg (φ_pet + φ_bg = f(картинка) − f(серое)). Для swapped фон = новая сцена Places365.
2. **Integrated Gradients** (ResNet-18 и методы, 500 тестовых картинок — те же, что для Grad-CAM):
   вклад каждого пикселя, суммируем по маске животного и по фону. Сравниваем с Grad-CAM из `05_analysis`.
3. Примеры карт IG для A и B, рисунки и таблицы → `report/figures/`, `results/analysis/` → push в GitHub.

Время на T4: ≈ 15 мин копирование + ≈ 15 мин Shapley + ≈ 25 мин IG.
'''),
    SETUP,
    code('''
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
'''),
    PROJECT,
    md("## 0. Checkpoints"),
    code('''
import shutil, zipfile
ARCHIVE = Path("/content/drive/MyDrive/dl/runs.zip")
CKPT = Path("/content/ckpt")
SEEDS = (0, 1, 2)
SHAPLEY_MODELS = ["resnet_A", "resnet_B", "resnet_gray", "resnet_blur", "resnet_cutmix", "resnet_rrr1",
                  "resnet_rrr10", "scratch_A", "scratch_B", "resnet50_A", "resnet50_B",
                  "convnext_A", "convnext_B", "swin_A", "swin_B"]
IG_MODELS = ["resnet_A", "resnet_B", "resnet_gray", "resnet_blur", "resnet_cutmix", "resnet_rrr1", "resnet_rrr10"]

members = {}
if ARCHIVE.exists():
    with zipfile.ZipFile(ARCHIVE) as z:
        for n in z.namelist():
            if n.endswith("/best.pt"):
                members[n.split("/")[-2]] = n
print(len(members), "best.pt in", ARCHIVE)

def checkpoint(run):
    dst = CKPT / run / "best.pt"
    if dst.exists():
        return dst
    if run in members:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE) as z, z.open(members[run]) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
        return dst
    if (RUNS / run / "best.pt").exists():
        return RUNS / run / "best.pt"
    return None

paths = {}
for m in SHAPLEY_MODELS:
    for s in SEEDS:
        p = checkpoint(f"{m}_seed{s}")
        if p is None:
            print("missing", f"{m}_seed{s}")
        else:
            paths[f"{m}_seed{s}"] = p
print(len(paths), "checkpoints ready")
'''),
    code('''
import numpy as np
import pandas as pd
from bgbias import analysis as A
from bgbias.attribution import RegionViews, integrated_gradients, region_shapley, split_by_mask
from bgbias.models import load_checkpoint
from bgbias.project import make_dataset
from bgbias.plotting import COLORS, INK2, saver, use_style
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

device = torch.device("cuda")
SPLIT = "test"
FIG_DIR = Path(REPO_DIR) / "report" / "figures"
TAB_DIR = Path(REPO_DIR) / "results" / "analysis"
OUT_DRIVE = DRIVE_ROOT / "results" / "attribution"
for d in (FIG_DIR, TAB_DIR, OUT_DRIVE):
    d.mkdir(parents=True, exist_ok=True)
use_style()
save = saver(FIG_DIR)

# all models stay on the GPU (~3 GB), so every image is rendered only once
models = {run: load_checkpoint(p, data.num_classes, device)[0] for run, p in paths.items()}
base = make_dataset(data, SPLIT, condition="swapped")
print(len(models), "models,", len(base), "test images")
'''),
    md('''
## 1. Region Shapley: вклад животного и фона (весь test)

φ_pet = ½[(v(P+B) − v(B)) + (v(P) − v(∅))], φ_bg = ½[(v(P+B) − v(P)) + (v(B) − v(∅))],
где v — log p(верный класс), P — животное, B — фон, ∅ — всё серое (как в fg-only / bg-only).
**bg_share** = |φ_bg| / (|φ_pet| + |φ_bg|) — какая доля решения приходится на фон.
**φ_bg < 0 на swapped** — новая сцена толкает модель от правильного ответа.
'''),
    code('''
from torch.utils.data import DataLoader

loader = DataLoader(RegionViews(base), batch_size=24, shuffle=False, num_workers=2, pin_memory=True)
per_image = {run: {c: {k: [] for k in ("phi_pet", "phi_bg", "f_full", "f_gray")} for c in ("original", "swapped")}
             for run in models}
for b, (views, mask, y) in enumerate(loader):
    views, y = views.to(device, non_blocking=True), y.to(device)
    for run, model in models.items():
        with torch.autocast("cuda", dtype=torch.float16):
            res = region_shapley(model, views, y)
        for c in res:
            for k, v in res[c].items():
                per_image[run][c][k].append(v.float().cpu())
    if b % 20 == 0:
        print(f"batch {b + 1}/{len(loader)}")

rows = []
for run, conds in per_image.items():
    for c, vals in conds.items():
        v = {k: torch.cat(x).numpy() for k, x in vals.items()}
        rows.append(pd.DataFrame({"run": run, "model": A.model_name(run), "condition": c,
                                  "image": np.arange(len(v["phi_pet"])), **v}))
shap_img = pd.concat(rows, ignore_index=True)
shap_img["bg_share"] = shap_img.phi_bg.abs() / (shap_img.phi_pet.abs() + shap_img.phi_bg.abs()).clip(lower=1e-8)
shap_img.to_csv(OUT_DRIVE / "region_shapley_per_image.csv.gz", index=False)

shap_run = (shap_img.groupby(["run", "model", "condition"])
            .agg(phi_pet=("phi_pet", "mean"), phi_bg=("phi_bg", "mean"), bg_share=("bg_share", "mean"),
                 bg_hurts=("phi_bg", lambda s: (s < 0).mean()), f_full=("f_full", "mean"))
            .reset_index())
shap_run.to_csv(TAB_DIR / "attribution_shapley_per_run.csv", index=False)
shap_model = shap_run.groupby(["model", "condition"])[["phi_pet", "phi_bg", "bg_share", "bg_hurts"]].agg(["mean", "std"])
shap_model.round(3)
'''),
    code('''
# A vs B: paired bootstrap over test images (per-image values averaged over the 3 seeds first)
def seed_mean(model, cond, col):
    d = shap_img[(shap_img.model == model) & (shap_img.condition == cond)]
    return d.groupby("image")[col].mean().to_numpy()

rng = np.random.default_rng(0)
boot_rows = []
pairs = list(A.FAMILIES.items()) + [(k, ("resnet_A", m)) for k, m in A.METHODS.items() if m not in ("resnet_A", "resnet_B")]
for fam, (ma, mb) in pairs:
    if not ((shap_img.model == ma).any() and (shap_img.model == mb).any()):
        continue
    for cond in ("original", "swapped"):
        for col in ("bg_share", "phi_bg"):
            d = seed_mean(ma, cond, col) - seed_mean(mb, cond, col)
            idx = rng.integers(0, len(d), size=(2000, len(d)))
            bs = d[idx].mean(1)
            boot_rows.append({"comparison": fam, "model_a": ma, "model_b": mb, "condition": cond, "metric": col,
                              "difference": d.mean(), "ci_low": np.quantile(bs, 0.025), "ci_high": np.quantile(bs, 0.975)})
shap_boot = pd.DataFrame(boot_rows)
shap_boot.to_csv(TAB_DIR / "attribution_shapley_bootstrap.csv", index=False)
shap_boot.round(4)
'''),
    code('''
# Figure: architectures, A vs B
fams = [f for f, (ma, mb) in A.FAMILIES.items() if (shap_run.model == ma).any() and (shap_run.model == mb).any()]
stat = shap_run.groupby(["model", "condition"])[["bg_share", "phi_bg"]].agg(["mean", "std"])
fig, axes = plt.subplots(1, 3, figsize=(14, 0.6 * len(fams) + 1.8), sharey=True)
y = np.arange(len(fams))
panels = ((axes[0], "original", "bg_share", "Share of the decision from the background\\n(original images)"),
          (axes[1], "swapped", "bg_share", "Share of the decision from the background\\n(swapped background)"),
          (axes[2], "swapped", "phi_bg", "Effect of the new background on log p(true class)\\n(swapped; < 0 = pushes away)"))
for ax, cond, col, title in panels:
    for variant, key, dy in (("A", "resnet_A", -0.12), ("B", "resnet_B", 0.12)):
        ms = [A.FAMILIES[f][0 if variant == "A" else 1] for f in fams]
        m = [stat.loc[(x, cond), (col, "mean")] for x in ms]
        s = [stat.loc[(x, cond), (col, "std")] for x in ms]
        ax.errorbar(m, y + dy, xerr=s, fmt="o", ms=7, color=COLORS[key], capsize=3,
                    label="A: original images" if variant == "A" else "B: background swap p = 0.5")
    ax.set_title(title)
    if col == "bg_share":
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    else:
        ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color="#ebeae6")
axes[0].set_yticks(y, fams)
axes[0].invert_yaxis()
axes[2].legend(loc="lower left")
fig.tight_layout()
save(fig, "fig10_attribution_architectures")
'''),
    md('''
## 2. Integrated Gradients (ResNet-18 и методы, 500 картинок)

IG_i = (x_i − x'_i) · среднее по 32 шагам ∂f/∂x_i на пути от серого x' к x. Сумма IG по маске животного
и по фону; **ig_on_pet** = доля |IG| на животном (сравнимо с Grad-CAM cam_on_pet).
completeness — |ΣIG − (f(x) − f(x'))|, проверка, что 32 шагов достаточно.
'''),
    code('''
rng = np.random.default_rng(0)
sample = np.sort(rng.choice(len(base), size=500, replace=False))   # same 500 images as Grad-CAM in 05
ig_views = RegionViews(base, indices=sample)
gray = ig_views[0][0][5].to(device)   # all-gray view = IG baseline (same for every image)
ig_runs = [r for r in models if A.model_name(r) in IG_MODELS]
rows = []
for k in range(len(ig_views)):
    views, mask, label = ig_views[k]
    views, mask = views.to(device), mask.to(device)
    for run in ig_runs:
        for cond, j in (("original", 0), ("swapped", 3)):
            attr, err = integrated_gradients(models[run], views[j], label, gray, steps=32, chunk=32)
            pet, bg, share = split_by_mask(attr, mask)
            rows.append({"run": run, "model": A.model_name(run), "condition": cond, "image": int(sample[k]),
                         "ig_pet": pet, "ig_bg": bg, "ig_on_pet": share, "completeness_err": err,
                         "pet_area": mask.mean().item()})
    if k % 50 == 0:
        print(f"image {k + 1}/{len(ig_views)}")
ig_img = pd.DataFrame(rows)
ig_img.to_csv(OUT_DRIVE / "integrated_gradients_per_image.csv.gz", index=False)
ig_run = ig_img.groupby(["run", "model", "condition"])[["ig_pet", "ig_bg", "ig_on_pet", "completeness_err", "pet_area"]].mean().reset_index()
ig_run.to_csv(TAB_DIR / "attribution_ig_per_run.csv", index=False)
ig_run.groupby(["model", "condition"])[["ig_pet", "ig_bg", "ig_on_pet", "completeness_err"]].agg(["mean", "std"]).round(3)
'''),
    code('''
# Figure: methods (ResNet-18): Shapley background effect and IG vs Grad-CAM share on the pet
cam = pd.read_csv(TAB_DIR / "gradcam_foreground_share.csv")
meth = [(label, m) for label, m in A.METHODS.items() if (ig_run.model == m).any()]
yy = np.arange(len(meth))
fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(meth) + 1.8), sharey=True)
for dy, cond, color in ((-0.12, "original", "#9a9893"), (0.12, "swapped", INK2)):
    d = shap_run[shap_run.condition == cond].groupby("model").phi_bg.agg(["mean", "std"])
    axes[0].errorbar([d.loc[m, "mean"] for _, m in meth], yy + dy, xerr=[d.loc[m, "std"] for _, m in meth],
                     fmt="o", ms=7, color=color, capsize=3, label=f"{cond} images")
axes[0].axvline(0, color=INK2, lw=0.8)
axes[0].set_title("Shapley: effect of the background on log p(true class)")
axes[0].legend(loc="lower left")
for dy, src, col, color, lab in ((-0.12, ig_run, "ig_on_pet", COLORS["resnet_B"], "Integrated Gradients"),
                                 (0.12, cam, "cam_on_pet", COLORS["resnet_A"], "Grad-CAM")):
    d = src[src.condition == "swapped"].groupby("model")[col].agg(["mean", "std"])
    axes[1].errorbar([d.loc[m, "mean"] for _, m in meth], yy + dy, xerr=[d.loc[m, "std"] for _, m in meth],
                     fmt="o", ms=7, color=color, capsize=3, label=lab)
area = ig_run.pet_area.mean()
axes[1].axvline(area, color=INK2, lw=0.8, ls="--")
axes[1].annotate("pet area", (area, len(meth) - 0.4), ha="center", fontsize=8, color=INK2)
axes[1].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[1].set_title("Share of attribution on the pet (swapped images)")
axes[1].legend(loc="lower right")
for ax in axes:
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, color="#ebeae6")
axes[0].set_yticks(yy, [label for label, _ in meth])
axes[0].invert_yaxis()
fig.tight_layout()
save(fig, "fig11_attribution_methods")
'''),
    md("## 3. Примеры: карты IG для A и B (seed 0)"),
    code('''
from scipy.ndimage import gaussian_filter
from bgbias.datasets import IMAGENET_MEAN, IMAGENET_STD

# images from the IG sample where the new scene hurts A most and B least
d = shap_img[(shap_img.condition == "swapped") & shap_img.image.isin(sample)]
a = d[d.run == "resnet_A_seed0"].set_index("image").phi_bg
b = d[d.run == "resnet_B_seed0"].set_index("image").phi_bg
pick = list((a - b).sort_values().index[:4])
pos = {int(i): k for k, i in enumerate(sample)}

def to_img(x):
    m, s = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()

def heat(attr):
    h = gaussian_filter(attr.abs().cpu().numpy(), 3)
    return h / max(h.max(), 1e-12)

breed = dict(zip(data.splits[SPLIT].label, data.splits[SPLIT].breed))
fig, axes = plt.subplots(len(pick), 6, figsize=(15, 2.6 * len(pick)))
for r, img_idx in enumerate(pick):
    views, mask, label = ig_views[pos[img_idx]]
    views = views.to(device)
    cols = []
    for j, cond in ((0, "original"), (3, "swapped")):
        cols.append((to_img(views[j]), None, cond))
        for run, key in (("resnet_A_seed0", "A"), ("resnet_B_seed0", "B")):
            attr, _ = integrated_gradients(models[run], views[j], label, gray, steps=32, chunk=32)
            with torch.no_grad():
                pred = models[run](views[j][None]).argmax(1).item()
            cols.append((to_img(views[j]), heat(attr), f"{key}: {breed[pred]}" + (" ✓" if pred == label else " ✗")))
    for c, (im, h, title) in enumerate(cols):
        ax = axes[r, c]
        ax.imshow(im)
        if h is not None:
            ax.imshow(h, cmap="inferno", alpha=0.55)
        ax.set_title(title, fontsize=8, color="#c2271d" if title.endswith("✗") else INK2)
        ax.axis("off")
    axes[r, 0].set_title(f"original · true: {breed[label]}", fontsize=8)
fig.tight_layout()
save(fig, "fig12_ig_examples")
'''),
    md("## Сохранение и push"),
    code('''
from bgbias.github_sync import sync

shutil.copytree(FIG_DIR, DRIVE_ROOT / "figures", dirs_exist_ok=True)
shutil.copytree(TAB_DIR, DRIVE_ROOT / "results" / "analysis", dirs_exist_ok=True)
sync(REPO_DIR, "Attribution: region Shapley (all models) and Integrated Gradients (ResNet-18)")
'''),
])

(OUT / "08_attribution.ipynb").write_text(json.dumps(attribution_nb, indent=1, ensure_ascii=False))
print("ok 08")


# ---------------------------------------------------------------- 09: mask robustness (paper plan, stage 1)
mask_nb = nb([
    md(f'''
# 09 · Mask robustness: does model B need perfect masks?
{colab_badge("09_mask_robustness.ipynb")}

**Runtime → T4 GPU → Run all.** ≈ 2.5–3 ч (4 конфигурации × 3 seed по ~10 мин + маски + оценка).
Повторный Run all безопасен: готовые маски, обученные запуски и предсказания берутся из кэша на Drive,
прерванное обучение продолжается с последней эпохи.

Модель B вставляет животное на новый фон по маске. В Oxford Pet маски ручные, в реальных данных их нет.
Обучаем B так же, как `resnet_B`, но с худшими масками **только для замены фона при обучении**:

| конфиг | маска при обучении |
|---|---|
| `resnet_B_deeplab` | автоматическая: DeepLabV3 (классы VOC cat + dog), без разметки |
| `resnet_B_dilate` | ручная, расширенная на 3% стороны фото (вокруг животного остаётся кусок старого фона) |
| `resnet_B_erode` | ручная, суженная на 3% (обрезан мех по краю) |
| `resnet_B_box` | прямоугольник вокруг животного (самая грубая маска) |

Тест не меняется: все условия (swapped, fg-only, bg-only) строятся по ручным маскам, поэтому цифры напрямую
сравнимы с `resnet_A` / `resnet_B` из основной таблицы (они берутся из `results/test/` в репо, не переобучаются).

Чекпоинты: `MyDrive/bgbias/runs_masks/` (отдельно от `runs/`). Маски: `MyDrive/bgbias/data/masks/`.
Результаты: `results/mask_robustness/`, рисунки `report/figures/fig13_mask_variants.png`,
`fig14_mask_robustness.png` → push в GitHub.
'''),
    SETUP,
    code('''
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
device = torch.device("cuda")
'''),
    PROJECT,
    md("## 1. Маски для обучения и их качество"),
    code('''
import numpy as np
import pandas as pd
from bgbias import masks as M
from bgbias.train import load_config

CONFIGS = ["resnet_B_deeplab", "resnet_B_dilate", "resnet_B_erode", "resnet_B_box"]
SEEDS = [0, 1, 2]
configs = {name: load_config(f"{REPO_DIR}/configs/{name}.yaml") for name in CONFIGS}

MASK_DIR = DRIVE_ROOT / "data" / "masks"
OUT = Path(REPO_DIR) / "results" / "mask_robustness"
OUT_DRIVE = DRIVE_ROOT / "results" / "mask_robustness"
for d in (OUT, OUT_DRIVE):
    d.mkdir(parents=True, exist_ok=True)

train_ids = data.splits["train"]["image_id"].tolist()
train_masks = {}
for name, cfg in configs.items():
    v, frac = cfg["train_mask"], cfg.get("mask_frac", 0.03)
    if v == "deeplab":
        train_masks[v] = M.build_deeplab(data.pet_root, train_ids, device, MASK_DIR / "train_deeplab.npz")
    else:
        train_masks[v] = M.build_perturbed(data.pet_root, train_ids, v, MASK_DIR / f"train_{v}_{frac}.npz", frac,
                                           cfg["include_border"])
    print(v, len(train_masks[v]), "masks")
'''),
    code('''
rows = []
for i in train_ids:
    ref = M.human_mask(data.pet_root, i)
    for v, packed in train_masks.items():
        q = M.quality(M.unpack_mask(packed[str(i)]), ref)
        rows.append({"image_id": i, "variant": v, **q, "used": q["area"] > 0.005})
quality = pd.DataFrame(rows)
quality_summary = quality.groupby("variant")[["iou", "recall", "precision", "area", "used"]].mean()
quality_summary = quality_summary.rename(columns={"used": "share_swapped"})   # empty masks -> photo kept
quality_summary.to_csv(OUT / "mask_quality.csv")
quality_summary.round(3)
'''),
    code('''
import matplotlib.pyplot as plt
from bgbias import compose
from bgbias.plotting import saver, use_style

use_style()
FIG_DIR = Path(REPO_DIR) / "report" / "figures"
save = saver(FIG_DIR)

# examples: median-IoU DeepLab image, a bad DeepLab image, two random ones
dl = quality[quality.variant == "deeplab"].set_index("image_id")["iou"]
rng = np.random.default_rng(0)
examples = [dl.sub(dl.median()).abs().idxmin(), dl.idxmin(), *rng.choice(train_ids, 2, replace=False)]
cols = ["human"] + list(train_masks)
fig, axes = plt.subplots(len(examples), len(cols), figsize=(2.4 * len(cols), 2.4 * len(examples)))
for r, image_id in enumerate(examples):
    img = np.asarray(compose.load_image(data.pet_root / "images" / f"{image_id}.jpg"), dtype=np.float32) / 255
    ref = M.human_mask(data.pet_root, image_id)
    for c, v in enumerate(cols):
        m = ref if v == "human" else M.unpack_mask(train_masks[v][str(image_id)])
        ax = axes[r, c]
        ax.imshow(img * (0.35 + 0.65 * m[..., None]))
        ax.contour(m, levels=[0.5], colors="#eb6834", linewidths=1.2)
        title = v if v == "human" else f"{v} · IoU {M.quality(m, ref)['iou']:.2f}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
fig.tight_layout()
save(fig, "fig13_mask_variants")
'''),
    md("## 2. Обучение (seed-major: если Colab оборвётся, у каждого варианта уже будет seed 0)"),
    code('''
from bgbias.train import prune_last_checkpoints, train

RUNS_M = DRIVE_ROOT / "runs_masks"
RUNS_M.mkdir(parents=True, exist_ok=True)
prune_last_checkpoints(RUNS_M)
histories = {}
for seed in SEEDS:
    for name, cfg in configs.items():
        run_dir = RUNS_M / f"{name}_seed{seed}"
        print(f"=== {run_dir.name} ===")
        histories[(name, seed)] = pd.DataFrame(train(cfg, seed, data, run_dir, device,
                                                     train_masks=train_masks[cfg["train_mask"]]))
'''),
    md("## 3. Оценка на test (ручные маски, как в основной таблице)"),
    code('''
from bgbias import analysis as A, evaluate as E
from bgbias.models import load_checkpoint

tables = {}
for name in CONFIGS:
    for seed in SEEDS:
        run = f"{name}_seed{seed}"
        cached = OUT_DRIVE / f"{run}_predictions.csv"
        if cached.exists():
            table = pd.read_csv(cached)
        else:
            model, cfg = load_checkpoint(RUNS_M / run / "best.pt", data.num_classes, device)
            predict = E.torch_predict_fn(model, data, "test", device, include_border=cfg["include_border"],
                                         blur_sigma=cfg["blur_sigma"], img_size=cfg["img_size"])
            table = E.predictions_table(data, "test", predict)
            table.to_csv(cached, index=False)
        table.to_csv(OUT / f"{run}_predictions.csv", index=False)
        tables[run] = table
# reference models from the main experiment (not retrained)
for ref in ("resnet_A", "resnet_B"):
    for seed in SEEDS:
        run = f"{ref}_seed{seed}"
        tables[run] = pd.read_csv(Path(REPO_DIR) / "results" / "test" / f"{run}_predictions.csv")

per_run = pd.DataFrame({run: E.summarize(t) for run, t in tables.items()}).T
per_run["model"] = [A.model_name(r) for r in per_run.index]
per_run.to_csv(OUT / "summary_per_run.csv")
ORDER = ["resnet_A", "resnet_B"] + CONFIGS
COLS = ["acc_original", "acc_swapped", "acc_fgonly", "acc_bgonly", "gap_swap", "flip_swap"]
summary = per_run.groupby("model")[COLS].agg(["mean", "std"]).loc[ORDER]
summary.round(4)
'''),
    code('''
# paired bootstrap over test images: is the gap of each mask variant different from B (human mask) and from A?
rows = []
for name in CONFIGS:
    for ref in ("resnet_B", "resnet_A"):
        b = A.bootstrap_gap_difference(tables, ref, name, n_boot=2000)
        rows.append({"variant": name, "reference": ref, "gap_ref": b["gap_A"], "gap_variant": b["gap_B"],
                     "difference": b["difference"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
boot = pd.DataFrame(rows)
boot.to_csv(OUT / "bootstrap_gap_vs_reference.csv", index=False)
boot.round(4)
'''),
    code('''
from bgbias.plotting import COLORS, INK2

labels = {"resnet_A": "A: no swap", "resnet_B": "B: human mask"}
for name, cfg in configs.items():
    labels[name] = f"B: {cfg['train_mask']}\\n(IoU {quality_summary.loc[cfg['train_mask'], 'iou']:.2f})"
x = np.arange(len(ORDER))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, col, title in ((axes[0], "gap_swap", "Accuracy drop when the background is swapped (p.p.)"),
                       (axes[1], "acc_original", "Accuracy on original images (%)")):
    mean, std = summary[(col, "mean")] * 100, summary[(col, "std")] * 100
    colors = [COLORS["resnet_A"] if m == "resnet_A" else COLORS["resnet_B"] for m in ORDER]
    ax.bar(x, mean, yerr=std, color=colors, capsize=3, width=0.65)
    for xi, m in zip(x, mean):
        ax.text(xi, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [labels[m] for m in ORDER], fontsize=8)
    ax.set_title(title)
axes[1].set_ylim(80, 95)
fig.tight_layout()
save(fig, "fig14_mask_robustness")
'''),
    md("## Сохранение и push"),
    code('''
import shutil
from bgbias.github_sync import sync

shutil.copytree(OUT, OUT_DRIVE, dirs_exist_ok=True)
shutil.copytree(FIG_DIR, DRIVE_ROOT / "figures", dirs_exist_ok=True)
sync(REPO_DIR, "Mask robustness: model B trained with DeepLab, dilated, eroded and box masks")
'''),
])

(OUT / "09_mask_robustness.ipynb").write_text(json.dumps(mask_nb, indent=1, ensure_ascii=False))
print("ok 09")


# ---------------------------------------------------------------- 10: cue or distractor (paper, controlled cue)
cue_nb = nb([
    md(f'''
# 10 · Cue or distractor? A controlled background cue
{colab_badge("10_cue_or_distractor.ipynb")}

**Runtime → T4 GPU → Run all.** ≈ 2–2.5 ч (3 конфигурации × 3 seed по ~10 мин + оценка ~20 мин).
Повторный Run all безопасен: обученные запуски пропускаются, прерванное обучение продолжается.

**Идея.** Модель может зависеть от фона двумя разными способами:
- **подсказка (cue)** — фон предсказывает породу, и модель этим пользуется (shortcut);
- **помеха (distractor)** — фон о породе ничего не говорит, но незнакомая сцена сбивает модель.

Наша ResNet-18 A похожа на помеху. Чтобы проверить, что диагностика различает эти случаи, мы **создаём подсказку
известной силы**: у каждой породы своя «домашняя» сцена (категория Places365). При замене фона (p = 0.5, как у B)
с вероятностью ρ новая сцена берётся из домашней категории породы.

| конфиг | ρ |
|---|---|
| `resnet_B` (уже обучена, не переобучается) | 0 |
| `cue_rho050` | 0.5 |
| `cue_rho090` | 0.9 |
| `cue_rho100` | 1.0 |

Для сравнения оцениваются также `resnet_A`, `scratch_A`, `scratch_B` (чекпоинты из `MyDrive/dl/runs.zip`).

**Тест** (ручные маски): обычные original / fg-only / bg-only / swapped + три новых условия:
`congruent` — животное на сцене своей домашней категории (новые, не виденные при обучении фото этой категории);
`conflict` — на домашней сцене **другой** породы; и их bg-only варианты.

**Диагностика:** cue score = точность по одному домашнему фону − случайная (1/37); conflict follow = доля ответов
«порода, чья это сцена»; distraction score = −φ_bg чужой сцены (Shapley, наты). Плюс доля Grad-CAM на животном.

Результаты: `results/cue/`, рисунок `report/figures/fig15_cue_vs_distractor.png` → push в GitHub.
'''),
    SETUP,
    code('''
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> T4 GPU"
device = torch.device("cuda")
'''),
    PROJECT,
    md("## 1. Домашние сцены и тестовый манифест"),
    code('''
import dataclasses
import numpy as np
import pandas as pd
from bgbias import cue as C

home_train, home_test, home_table = C.make_home_scenes(data.pools, data.num_classes, seed=0)
data_cue = dataclasses.replace(data, pools=C.pools_without_heldout(data.pools, home_test))
manifest = C.make_cue_manifest(data.splits["test"], home_test, seed=1)

OUT = Path(REPO_DIR) / "results" / "cue"
OUT_DRIVE = DRIVE_ROOT / "results" / "cue"
for d in (OUT, OUT_DRIVE):
    d.mkdir(parents=True, exist_ok=True)
breeds = data.splits["train"].drop_duplicates("label").set_index("label")["breed"].sort_index()
home_table["breed"] = home_table["label"].map(breeds)
home_table.to_csv(OUT / "home_scenes.csv", index=False)
manifest.to_csv(OUT / "test_manifest.csv", index=False)
print(len(data.bg_pool("train")), "->", len(data_cue.bg_pool("train")), "random training scenes (held-out removed)")
home_table.head(10)
'''),
    md("## 2. Обучение (seed-major)"),
    code('''
from bgbias.train import load_config, prune_last_checkpoints, train

CONFIGS = ["cue_rho050", "cue_rho090", "cue_rho100"]
SEEDS = [0, 1, 2]
configs = {name: load_config(f"{REPO_DIR}/configs/{name}.yaml") for name in CONFIGS}
RUNS_C = DRIVE_ROOT / "runs_cue"
RUNS_C.mkdir(parents=True, exist_ok=True)
prune_last_checkpoints(RUNS_C)
for seed in SEEDS:
    for name, cfg in configs.items():
        run_dir = RUNS_C / f"{name}_seed{seed}"
        print(f"=== {run_dir.name} ===")
        train(cfg, seed, data_cue, run_dir, device, home_bgs=home_train)
'''),
    md("## 3. Чекпоинты для оценки"),
    code('''
import shutil, zipfile
ARCHIVE = Path("/content/drive/MyDrive/dl/runs.zip")
CKPT = Path("/content/ckpt")
REFERENCE = ["resnet_A", "resnet_B", "scratch_A", "scratch_B"]

members = {}
if ARCHIVE.exists():
    with zipfile.ZipFile(ARCHIVE) as z:
        members = {n.split("/")[-2]: n for n in z.namelist() if n.endswith("/best.pt")}

def checkpoint(run):
    if (RUNS_C / run / "best.pt").exists():
        return RUNS_C / run / "best.pt"
    dst = CKPT / run / "best.pt"
    if not dst.exists() and run in members:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE) as z, z.open(members[run]) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
    return dst if dst.exists() else None

paths = {}
for m in REFERENCE + CONFIGS:
    for s in SEEDS:
        p = checkpoint(f"{m}_seed{s}")
        if p is None:
            print("missing", f"{m}_seed{s}")
        else:
            paths[f"{m}_seed{s}"] = p
print(len(paths), "checkpoints")
'''),
    md("## 4. Оценка: 10 вариантов каждой тестовой картинки, все модели за один проход"),
    code('''
from torch.utils.data import DataLoader
from bgbias import analysis as A
from bgbias.models import load_checkpoint
from bgbias.project import make_dataset

models = {run: load_checkpoint(p, data.num_classes, device)[0] for run, p in paths.items()}
views_ds = C.CueViews(make_dataset(data, "test", condition="swapped"), manifest)
loader = DataLoader(views_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

parts = {run: [] for run in models}
for b, (views, mask, y, y_conf) in enumerate(loader):
    views, yd, ycd = views.to(device, non_blocking=True), y.to(device), y_conf.to(device)
    for run, model in models.items():
        with torch.autocast("cuda", dtype=torch.float16):
            pred, lp, lpc = C.view_outputs(model, views, yd, ycd)
        parts[run].append(C.per_image_table(pred.cpu(), lp.float().cpu(), lpc.float().cpu(), y, y_conf))
    if b % 25 == 0:
        print(f"batch {b + 1}/{len(loader)}")

per_image = {run: pd.concat(t, ignore_index=True) for run, t in parts.items()}
pd.concat([t.assign(run=r) for r, t in per_image.items()]).to_csv(OUT_DRIVE / "per_image.csv.gz", index=False)
per_run = pd.DataFrame({run: C.summarize(t) for run, t in per_image.items()}).T
per_run["model"] = [A.model_name(r) for r in per_run.index]
per_run.to_csv(OUT / "summary_per_run.csv")
ORDER = ["resnet_A", "scratch_A", "scratch_B", "resnet_B"] + CONFIGS
KEY = ["acc_original", "acc_swapped", "acc_congruent", "acc_conflict", "acc_bgonly", "acc_congruent_bgonly",
       "follow_conflict", "phi_bg_own", "phi_bg_foreign", "phi_bg_congruent", "phi_bg_conflict",
       "cue_score", "distraction_score"]
summary = per_run.groupby("model")[KEY].agg(["mean", "std"]).reindex(ORDER)
summary.to_csv(OUT / "summary.csv")
summary.xs("mean", axis=1, level=1).round(3)
'''),
    md("## 5. Grad-CAM: видит ли тепловая карта разницу? (500 тестовых картинок)"),
    code('''
from bgbias.gradcam import GradCAM

rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(views_ds), 500, replace=False))
sub = torch.utils.data.Subset(views_ds, idx)
gl = DataLoader(sub, batch_size=16, shuffle=False, num_workers=2)
use = {v: C.VIEWS.index(v) for v in ("original", "congruent", "swapped")}
cam_rows = []
for run, model in models.items():
    cam = GradCAM(model)
    shares = {v: [] for v in use}
    for views, mask, y, _ in gl:
        m = mask.to(device)
        for v, j in use.items():
            c, _ = cam(views[:, j].to(device))
            shares[v].append(((c * m).flatten(1).sum(1) / c.flatten(1).sum(1).clamp_min(1e-8)).cpu())
    cam.remove()
    cam_rows.append({"run": run, "model": A.model_name(run),
                     **{f"cam_on_pet_{v}": torch.cat(s).mean().item() for v, s in shares.items()}})
cams = pd.DataFrame(cam_rows)
cams.to_csv(OUT / "gradcam_per_run.csv", index=False)
cams.groupby("model").mean(numeric_only=True).reindex(ORDER).round(3)
'''),
    md("## 6. Рисунок: карта «подсказка × помеха»"),
    code('''
import matplotlib.pyplot as plt
from bgbias.plotting import INK2, saver, use_style

use_style()
FIG_DIR = Path(REPO_DIR) / "report" / "figures"
save = saver(FIG_DIR)
NAMES = {"resnet_A": "A (no swap)", "scratch_A": "scratch A", "scratch_B": "scratch B", "resnet_B": "B, ρ = 0",
         "cue_rho050": "ρ = 0.5", "cue_rho090": "ρ = 0.9", "cue_rho100": "ρ = 1.0"}
COL = {"resnet_A": "#2a78d6", "scratch_A": "#7aa6e0", "scratch_B": "#f0a27f", "resnet_B": "#eb6834",
       "cue_rho050": "#b58cd6", "cue_rho090": "#8a55c2", "cue_rho100": "#5b2a9a"}
models_present = [m for m in ORDER if m in summary.index and not summary.loc[m].isna().all()]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
ax = axes[0]
for m in models_present:
    r = summary.loc[m]
    ax.errorbar(r[("distraction_score", "mean")], r[("cue_score", "mean")] * 100,
                xerr=r[("distraction_score", "std")], yerr=r[("cue_score", "std")] * 100,
                fmt="o", ms=8, color=COL[m], capsize=3)
    ax.annotate(NAMES[m], (r[("distraction_score", "mean")], r[("cue_score", "mean")] * 100),
                textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK2)
ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("distraction: −φ_bg of a foreign scene (nats)")
ax.set_ylabel("cue: home-scene-only accuracy − chance (p.p.)")
ax.set_title("Cue vs distractor")

ax = axes[1]
x = np.arange(len(models_present))
for k, (col, lab) in enumerate((("follow_conflict", "conflict: predicts the scene's breed"),
                                ("gap_conflict", "conflict: accuracy drop"))):
    if col not in per_run:
        continue
    vals = per_run.groupby("model")[col].agg(["mean", "std"]).reindex(models_present) * 100
    ax.bar(x + (k - 0.5) * 0.38, vals["mean"], 0.38, yerr=vals["std"], capsize=2,
           color=["#5b2a9a", "#bbb"][k], label=lab)
ax.set_xticks(x, [NAMES[m] for m in models_present], rotation=30, ha="right", fontsize=8)
ax.set_title("Pet on another breed's home scene (%)"); ax.legend(fontsize=8)

ax = axes[2]
cm = cams.groupby("model")[["cam_on_pet_original", "cam_on_pet_congruent"]].agg(["mean", "std"]).reindex(models_present)
for k, v in enumerate(("original", "congruent")):
    ax.bar(x + (k - 0.5) * 0.38, cm[(f"cam_on_pet_{v}", "mean")] * 100, 0.38,
           yerr=cm[(f"cam_on_pet_{v}", "std")] * 100, capsize=2, color=["#999", "#5b2a9a"][k], label=v)
ax.set_ylim(50, 100)
ax.set_xticks(x, [NAMES[m] for m in models_present], rotation=30, ha="right", fontsize=8)
ax.set_title("Grad-CAM mass on the pet (%)"); ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "fig15_cue_vs_distractor")
'''),
    md("## Сохранение и push"),
    code('''
from bgbias.github_sync import sync

shutil.copytree(OUT, OUT_DRIVE, dirs_exist_ok=True)
shutil.copytree(FIG_DIR, DRIVE_ROOT / "figures", dirs_exist_ok=True)
sync(REPO_DIR, "Cue or distractor: controlled background cue (rho = 0.5, 0.9, 1.0) and the diagnostic")
'''),
])

(OUT / "10_cue_or_distractor.ipynb").write_text(json.dumps(cue_nb, indent=1, ensure_ascii=False))
print("ok 10")
