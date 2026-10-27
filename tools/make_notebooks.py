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

Checkpoints: `MyDrive/bgbias/runs/<name>_seed<k>/` (`best.pt` по val original, `last.pt` для продолжения).
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
