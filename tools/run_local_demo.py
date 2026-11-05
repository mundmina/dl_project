"""Run 06_demo with a persistent local Jupyter kernel and saved outputs."""
import json
import os
from pathlib import Path
import time

import run_local_batch as batch


def main():
    nb = batch.prepare("06_demo")
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if '"pip", "install"' in cell.source:
            cell.source = 'import torch\nassert torch.cuda.is_available(), "CUDA is required"'
        cell.source = cell.source.replace(
            "demo.launch(share=True)",
            'demo.launch(share=False, server_name="127.0.0.1", server_port=7860, '
            'prevent_thread_lock=True, inline=False, '
            'allowed_paths=[str(data.pet_root / "images"), str(data.places_root)])')
    destination = batch.OUT / "06_demo.ipynb"
    status_path = batch.OUT / "demo_status.json"
    state = {"pid": os.getpid(), "state": "starting", "url": "http://127.0.0.1:7860"}
    status_path.write_text(json.dumps(state, indent=2))
    client = batch.NotebookClient(nb, timeout=600, startup_timeout=120, kernel_name="python3")
    try:
        with client.setup_kernel(cwd=str(batch.REPO / "notebooks")):
            for index, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                print("Executing demo cell", index, flush=True)
                try:
                    client.execute_cell(cell, index)
                finally:
                    batch.nbformat.write(nb, destination)
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        print(output.text, end="", flush=True)
            state["state"] = "running"
            status_path.write_text(json.dumps(state, indent=2))
            print("Demo ready:", state["url"], flush=True)
            while True:
                time.sleep(10)
    except Exception as error:
        state.update(state="failed", error=str(error))
        status_path.write_text(json.dumps(state, indent=2))
        raise


if __name__ == "__main__":
    main()
