"""Copy small result files from Drive into the repo and push them to GitHub from Colab.

Needs a Colab secret GITHUB_TOKEN: fine-grained token, Contents read/write on mundmina/dl_project.
Large files (datasets, checkpoints, HOG features) never go to git.
"""
import shutil
import subprocess
from pathlib import Path

from .utils import DRIVE_ROOT

REMOTE = "github.com/mundmina/dl_project.git"
GIT_NAME, GIT_EMAIL = "mundmina", "mundmina@users.noreply.github.com"
RESULT_PATHS = ("splits", "manifests", "results", "report/figures")


def _copy(src, dst):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy(src, dst)


def collect_results(repo_dir, drive_root=DRIVE_ROOT):
    """Drive -> repo: week-1 data outputs, val/test metrics and predictions, per-run histories."""
    repo_dir, drive_root = Path(repo_dir), Path(drive_root)
    week1 = drive_root / "week1"
    for rel in ("splits", "manifests", "results/data_audit.csv", "report/figures/week1_gallery.jpg"):
        _copy(week1 / rel, repo_dir / rel)
    for split in ("val", "test"):
        _copy(drive_root / "results" / split, repo_dir / "results" / split)
    for run in sorted((drive_root / "runs").glob("*")):
        for name in ("history.json", "config.yaml", "val_metrics.json", "c_grid.csv"):
            _copy(run / name, repo_dir / "results" / "runs" / run.name / name)


def push(repo_dir, message, paths=RESULT_PATHS, branch="main"):
    """Commit `paths` and push to GitHub. Prints a note and returns if no token is configured."""
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
    except Exception as e:  # noqa: BLE001 - missing secret, access not granted, not in Colab
        print(f"skip push to GitHub ({type(e).__name__}): add the GITHUB_TOKEN secret in Colab")
        return
    token = token.strip()
    if token.count("github_pat_") > 1:
        print("skip push to GitHub: GITHUB_TOKEN contains several tokens pasted together, keep only one")
        return

    def git(*args, check=True):
        r = subprocess.run(["git", "-C", str(repo_dir), "-c", f"user.name={GIT_NAME}",
                            "-c", f"user.email={GIT_EMAIL}", *args], capture_output=True, text=True)
        out = (r.stdout + r.stderr).replace(token, "***").strip()
        if out:
            print(out)
        if check and r.returncode:
            # not check_returncode(): its traceback would print the command with the token in the URL
            raise RuntimeError(f"git {args[0]} failed (exit {r.returncode}), see output above")
        return r

    url = f"https://{GIT_NAME}:{token}@{REMOTE}"
    existing = [p for p in paths if (Path(repo_dir) / p).exists()]
    git("add", "--", *existing)
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("nothing new to commit")  # still push: an earlier failed push may have left a local commit
    else:
        git("commit", "-m", message)
    git("pull", "--rebase", url, branch)
    git("push", url, f"HEAD:{branch}")
    print("pushed to GitHub")


def sync(repo_dir, message):
    collect_results(repo_dir)
    push(repo_dir, message)
