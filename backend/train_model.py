#!/usr/bin/env python3
"""Train SuccessScoreMLP on global_startup_success_dataset.csv (notebook pipeline)."""

from __future__ import annotations

import argparse

from ml.training import train_success_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train startup success MLP model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    result = train_success_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print("Saved model to:", result["output_dir"])
    print("Features:", result["feature_columns"])
    print("Training samples:", result["training_samples"])
    print("Final validation MSE:", f"{result['final_val_mse']:.4f}")
    print("Device:", result["device"])


if __name__ == "__main__":
    main()