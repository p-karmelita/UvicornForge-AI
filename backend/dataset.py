"""Utility script: verify the local startup dataset is available."""

from pathlib import Path

from ml.dataset import get_dataset_info

if __name__ == "__main__":
    info = get_dataset_info()
    print("Dataset loaded:", info["loaded"])
    print("Dataset path:", info["path"])
    print("Rows:", info["rows"])
    if not info["loaded"]:
        expected = Path(__file__).resolve().parent / "global_startup_success_dataset.csv"
        raise SystemExit(f"Missing dataset. Place global_startup_success_dataset.csv in {expected.parent}")