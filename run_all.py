"""
run_all.py
----------
Reproduces the full pipeline end-to-end: preprocess -> train -> evaluate.
Run this from the project root:

    python run_all.py
"""

import subprocess
import sys
import os

STEPS = [
    ("Preprocessing", os.path.join("src", "preprocess.py")),
    ("Training models", os.path.join("src", "train.py")),
    ("Evaluating models", os.path.join("src", "evaluate.py")),
]


def main():
    for label, script in STEPS:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\n{label} failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print("\nPipeline complete. Check outputs/ for models, figures, and metrics.")


if __name__ == "__main__":
    main()
