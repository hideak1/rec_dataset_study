"""Execute a Jupyter notebook directly using nbclient."""
import sys
import time
import nbformat
from nbclient import NotebookClient

def run_notebook(notebook_path: str, timeout: int = 7200):
    path = notebook_path
    print(f"\n{'='*60}")
    print(f"Running: {path}")
    print(f"{'='*60}")

    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name='python3',
        resources={'metadata': {'path': str(__import__('pathlib').Path(path).parent)}},
    )

    start = time.time()
    try:
        client.execute()
        elapsed = time.time() - start
        print(f"SUCCESS in {elapsed:.0f}s")

        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"FAILED after {elapsed:.0f}s")
        print(f"ERROR: {str(e)[-2000:]}")

        # Save partial results
        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_nb_direct.py <notebook.ipynb> [timeout]")
        sys.exit(1)

    nb_path = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 7200
    ok = run_notebook(nb_path, timeout)
    sys.exit(0 if ok else 1)
