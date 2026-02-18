from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np

from hh_salary_model.model import LinearRegressionModel

EXIT_OK: Final[int] = 0
EXIT_USAGE: Final[int] = 2
EXIT_NOT_FOUND: Final[int] = 3
EXIT_PREDICT: Final[int] = 4
EXIT_UNEXPECTED: Final[int] = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hh-salary-model",
        description="Predict salaries (RUB) from x_data.npy using saved linear regression weights.",
    )
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "resources" / "model.npz",
        help="Path to model weights (.npz). Default: resources/model.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file (preds.json / preds.txt). If omitted, prints JSON list to stdout.",
    )
    return parser.parse_args()


def _save_predictions(pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_f64 = pred.astype(np.float64)
    suffix = output_path.suffix.lower()

    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(pred_f64.tolist(), f, ensure_ascii=False, indent=2)
        return

    # txt/csv: one value per line
    np.savetxt(output_path, pred_f64, fmt="%.6f")


def _validate_paths(x_path: Path, model_path: Path) -> None:
    if not x_path.exists():
        raise FileNotFoundError(f"x_data.npy not found: {x_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            f"Train the model first: python scripts/train.py X.npy Y.npy --out resources/model.npz"
        )


def _predict(x_path: Path, model_path: Path) -> np.ndarray:
    x = np.load(x_path)
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError(f"x_data.npy must be a 2D numpy array, got: {getattr(x, 'shape', None)}")

    model = LinearRegressionModel.load(model_path)
    return model.predict(x)


def run(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args() if argv is None else _parse_args_from(argv)
        _validate_paths(args.x_path, args.model)
        pred = _predict(args.x_path, args.model)

        if args.output is None:
            # Requirement: return list of float (RUB) to stdout
            print(json.dumps(pred.astype(float).tolist(), ensure_ascii=False))
            return EXIT_OK

        _save_predictions(pred, args.output)
        print(f"Saved {len(pred)} predictions to: {args.output}")
        return EXIT_OK

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return EXIT_NOT_FOUND
    except (ValueError, np.linalg.LinAlgError) as e:
        # invalid input array / shape mismatch / numerical error
        print(f"ERROR: {e}")
        return EXIT_PREDICT
    except SystemExit as e:
        # argparse may raise SystemExit on bad args
        code = int(e.code) if isinstance(e.code, int) else EXIT_USAGE
        return code if code != 0 else EXIT_USAGE
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return EXIT_UNEXPECTED


def _parse_args_from(argv: list[str]) -> argparse.Namespace:
    # helper to make run(argv) testable without touching sys.argv
    parser = argparse.ArgumentParser(
        prog="hh-salary-model",
        description="Predict salaries (RUB) from x_data.npy using saved linear regression weights.",
    )
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "resources" / "model.npz",
        help="Path to model weights (.npz). Default: resources/model.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file (preds.json / preds.txt). If omitted, prints JSON list to stdout.",
    )
    return parser.parse_args(argv)


def main() -> int:
    # keep a main() with stable signature for "python -m" or app wrapper
    return run()
