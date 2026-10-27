# Running the notebooks on your own GPU (instead of Colab)

The same notebooks run in local Jupyter. Checkpoints still go to the shared `bgbias` Drive folder
(through Google Drive for desktop), so `04`/`05` in Colab later see your models, and results are pushed to GitHub.

Needs an NVIDIA GPU with ≥ 8 GB memory. One notebook per person: never run the same `03*` notebook
in two places at once, they would overwrite each other's checkpoints.

## Once

1. Install [Git](https://git-scm.com/downloads), [Miniconda](https://docs.anaconda.com/miniconda/) (Python 3.11+)
   and [Google Drive for desktop](https://www.google.com/drive/download/). Sign in to Drive with the account
   where `bgbias` was added to My Drive (Shared with me → `bgbias` → Organize → Add shortcut → My Drive).
2. Check that `G:\My Drive\bgbias` (Windows) or `~/Google Drive/My Drive/bgbias` (macOS) exists and shows `runs`, `data`, `week1`.
3. Clone and install (Anaconda Prompt / terminal):
   ```
   git clone https://github.com/mundmina/dl_project.git
   cd dl_project
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt pandas scikit-learn matplotlib jupyterlab
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   The last line must print `True` and the GPU name.
4. GitHub token: classic token at <https://github.com/settings/tokens/new>, scope **repo**, 90 days
   (you must be a collaborator on the repo). Keep it to yourself; it only goes into the variable below.

## Every time

Windows (PowerShell):
```
cd dl_project\notebooks
$env:BGBIAS_DRIVE = "G:\My Drive\bgbias"
$env:BGBIAS_DATA = "C:\bgbias_data"
$env:GITHUB_TOKEN = "<your token>"
jupyter lab
```
macOS / Linux:
```
cd dl_project/notebooks
export BGBIAS_DRIVE="$HOME/Google Drive/My Drive/bgbias" BGBIAS_DATA="$HOME/bgbias_data" GITHUB_TOKEN="<your token>"
jupyter lab
```
Open the notebook (e.g. `03d_train_resnet50_scratch.ipynb`) and Run all. The first cell must print the code
version and `GITHUB_TOKEN: ok`. The first run copies the dataset archives from Drive (~1.3 GB) to `BGBIAS_DATA`.
If it stops, Run all again: training resumes from the last epoch.
