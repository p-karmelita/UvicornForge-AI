#!/usr/bin/env python3
"""Evaluate exported SuccessScoreMLP against the validation split."""

from __future__ import annotations

import json

from ml.evaluation import evaluate_saved_model


def main() -> None:
    result = evaluate_saved_model()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()