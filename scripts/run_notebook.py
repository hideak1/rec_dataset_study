"""Execute a Jupyter notebook and save the output in-place."""
import sys
import json
import subprocess
import time
from pathlib import Path


def run_notebook(notebook_path: str, timeout: int = 3600):
    """Execute a notebook using jupyter nbconvert."""
    path = Path(notebook_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {path.name}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout", str(timeout),
            "--ExecutePreprocessor.kernel_name", "python3",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 60,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"SUCCESS in {elapsed:.0f}s")
        return True
    else:
        print(f"FAILED after {elapsed:.0f}s")
        print(f"STDERR: {result.stderr[-2000:]}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook.ipynb> [timeout_seconds]")
        sys.exit(1)

    nb_path = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    ok = run_notebook(nb_path, timeout)
    sys.exit(0 if ok else 1)
