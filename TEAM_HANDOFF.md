# Completed experiments and shared model files

The local run completed on 5 November 2026. Results include 51 Oxford-IIIT Pet model runs,
four-condition Pet evaluation, Stanford Dogs external evaluation (3,730 photos across 21 shared
breeds), analysis and background probing, plus six Waterbirds runs and test evaluation.
The interactive demo was tested with photo upload, gallery background selection, background
replacement, and predictions from both ResNet-18 models.

## Code and results

- `results/test/`: Pet predictions and summary tables.
- `results/stanford_dogs/`: external predictions and summaries.
- `results/analysis/`: statistical comparisons, Grad-CAM, architecture/method comparisons and probing.
- `results/waterbirds/`: test metrics and training histories for the Waterbirds case study.
- `results/runs/`: per-run configurations and training histories.
- `report/figures/`: report figures 1–9 and the data gallery.

Datasets and `.pt` model files are stored separately from Git. HOG-SVM predictions are included;
the original HOG-SVM `model.joblib` was not available on this machine.

## Shared ZIP files

Extract the three archives into one folder called `bgbias`:

```text
bgbias/
  runs/                 # 51 Pet runs: best.pt, last.pt, config.yaml, history.json
  runs_waterbirds/      # 6 Waterbirds runs: best.pt, last.pt, history.json
  executed/             # executed notebooks, logs and demo verification screenshot
```

The archives retain both selected (`best.pt`) and resumable (`last.pt`) checkpoints.
For an existing shared Drive folder, merge the extracted folders and keep a backup of any
older files that need to be retained. Dataset archives are not part of these three ZIP files.

## Running again

Follow `LOCAL.md` for the GPU Python environment, or open the notebooks in Colab using the
README links. Point `BGBIAS_DRIVE` at the extracted `bgbias` folder and `BGBIAS_DATA` at your
local datasets. `BGBIAS_WORKERS=2` was used for this Windows run.

The notebooks download datasets when needed. If Stanford downloads time out,
`tools/recover_external_data.py` contains the mirrors used for this run: the
`jessicali9530/stanford-dogs-dataset` Kaggle dataset and `grodino/waterbirds` on Hugging Face.
The recovery validates image/annotation counts and checks Waterbirds file hashes, metadata,
labels and official splits. The local helper scripts currently use the Windows paths
`C:/bgbias_drive`, `C:/bgbias_data` and the current Python interpreter; adjust the path constants
before using them on another machine. The mirror recovery also requires `pyarrow`.

Run `06_demo.ipynb` for the interactive demo; it needs `resnet_A_seed0/best.pt` and
`resnet_B_seed0/best.pt`. The local launcher `tools/run_local_demo.py` requires `gradio` and
serves the app at `http://127.0.0.1:7860`. That address works only on the computer running it.

The executed notebook copies record local execution adjustments. Git pulls and GitHub publishing
were disabled during the batch. Original training configurations, epochs and seed lists were retained.
