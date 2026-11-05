"""Execute the remaining notebooks locally; never pull, commit, or push code."""
import ctypes
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("C:/bgbias_drive")
OUT = ROOT / "executed" / "remaining_batch"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(BGBIAS_DRIVE=str(ROOT), BGBIAS_DATA="C:/bgbias_data",
                  BGBIAS_WORKERS="2", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
                  PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
os.environ.pop("GITHUB_TOKEN", None)
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]

import nbformat
from nbclient import NotebookClient

QUEUE = ["03c_train_convnext_swin", "03_train_resnet", "03b_train_methods",
         "04_evaluate", "04b_external_stanford_dogs",
         "05_analysis", "05b_probing", "07_case_waterbirds"]
STATE = {"pid": os.getpid(), "started": datetime.datetime.now().isoformat(), "notebooks": {}}


def status():
    temp = OUT / "status.tmp"
    temp.write_text(json.dumps(STATE, indent=2), encoding="utf-8")
    temp.replace(OUT / "status.json")


SETUP = f'''import os, sys
REPO_DIR = {str(REPO)!r}
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import torch
torch.set_num_threads(4)
from bgbias import github_sync
def local_sync(repo_dir, message):
    github_sync.collect_results(repo_dir)
    print("Saved locally:", message)
github_sync.sync = local_sync
print("Local execution; repository updates and GitHub publishing disabled")
'''


def prepare(name):
    nb = nbformat.read(REPO / "notebooks" / (name + ".ipynb"), as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if "# Shared setup:" in source:
            cell.source = SETUP
        # Evaluation must retain previous models whose checkpoints are on another machine.
        if name == "04_evaluate" and source.startswith("per_run = pd.DataFrame(summaries).T"):
            cell.source = '''for prediction_file in OUT.glob("*_predictions.csv"):
    summaries[prediction_file.name.removesuffix("_predictions.csv")] = E.summarize(pd.read_csv(prediction_file))
''' + source
        if name == "05_analysis" and 'models = {m: load_checkpoint' in source:
            if not all((ROOT / "runs" / f"{m}_seed0" / "best.pt").exists() for m in ("resnet_A", "resnet_B")):
                cell.source = 'print("Skipped flip-example Grad-CAM: original A/B checkpoints missing; existing figure retained")'
        if name == "05_analysis" and "cam_stats = pd.DataFrame(rows)" in source:
            cell.source = source.replace("cam_stats = pd.DataFrame(rows)", '''cam_stats = pd.DataFrame(rows)
previous = TAB_DIR / "gradcam_foreground_share.csv"
if previous.exists():
    old = pd.read_csv(previous)
    cam_stats = pd.concat([old[~old.model.isin(cam_stats.model)], cam_stats], ignore_index=True)''')
    nb.metadata["local_batch"] = {"source": str(REPO / "notebooks" / (name + ".ipynb")),
                                  "notes": "Local setup, no Git operations; retain unavailable baseline results."}
    return nb


def main():
    backup = ROOT / "backups" / "before_baseline_retraining"
    if not backup.exists():
        backup.mkdir(parents=True)
        for relative in ("results", "report/figures"):
            shutil.copytree(REPO / relative, backup / relative, dirs_exist_ok=True)
        print("Previous results backed up to", backup, flush=True)
    # Keep Windows awake while work is active; the display may still turn off.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for name in QUEUE:
            entry = STATE["notebooks"][name] = {"state": "running", "started": datetime.datetime.now().isoformat()}
            status()
            print("START", name, flush=True)
            nb = prepare(name)
            destination = OUT / (name + ".ipynb")

            def started(cell, cell_index, **kwargs):
                entry["cell"] = cell_index
                status()
                print(name, "cell", cell_index, flush=True)

            def saved(cell, cell_index, **kwargs):
                nbformat.write(nb, destination)
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        print(output.text, end="", flush=True)

            try:
                NotebookClient(nb, timeout=None, kernel_name="python3", startup_timeout=120,
                               resources={"metadata": {"path": str(REPO / "notebooks")}},
                               on_cell_execute=started, on_cell_executed=saved).execute()
                entry["state"] = "completed"
            except Exception as error:
                entry.update(state="failed", error=str(error)[-6000:])
                traceback.print_exc()
            finally:
                nbformat.write(nb, destination)
                entry["finished"] = datetime.datetime.now().isoformat()
                status()
                print(entry["state"].upper(), name, flush=True)
        STATE["finished"] = datetime.datetime.now().isoformat()
        status()
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
