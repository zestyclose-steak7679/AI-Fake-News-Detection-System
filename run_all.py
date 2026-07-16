import os
import sys

def main():
    print("="*50)
    print("Starting Fake News Detection Pipeline")
    print("="*50)

    scripts = [
        ("Preprocessing", "src/preprocess.py"),
        ("Feature Extraction", "src/features.py"),
        ("Model Training", "src/train.py"),
        ("Evaluation", "src/evaluate.py")
    ]

    for name, script in scripts:
        print(f"\n[{name}] Running {script}...")

        # Using sys.executable ensures the script uses the same python binary as run_all.py
        result = os.system(f"{sys.executable} {script}")

        if result != 0:
            print(f"Error: {script} failed. Halting pipeline.")
            sys.exit(1)

    print("\n" + "="*50)
    print("Pipeline completed successfully!")
    print("Check the 'outputs/' directory for results.")
    print("="*50)

if __name__ == "__main__":
    main()