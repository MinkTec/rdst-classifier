"""Export fitted aeon RDSTClassifier models to rdst-classifier JSON.

The Rust crate intentionally consumes a small, explicit JSON format instead of
Python pickles. This module extracts the fitted aeon transformer, scaler, and
ridge classifier parameters needed by the Rust inference path.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def export_rdst_classifier(clf: Any, *, version: str = "1.0") -> dict[str, Any]:
    """Return a JSON-serialisable model dict from a fitted aeon RDSTClassifier.

    This supports aeon's default RDSTClassifier estimator pipeline:
    RandomDilatedShapeletTransform -> StandardScaler -> RidgeClassifierCV.
    Custom estimators are intentionally rejected unless they expose the same
    fitted scaler and ridge attributes.
    """

    transformer = getattr(clf, "_transformer", None)
    estimator = getattr(clf, "_estimator", None)
    if (
        transformer is None
        or estimator is None
        or not hasattr(transformer, "shapelets_")
    ):
        raise ValueError("RDSTClassifier must be fitted before export")

    scaler, ridge = _extract_default_estimator_parts(estimator)

    (
        values,
        _startpoints,
        lengths,
        dilations,
        thresholds,
        normalises,
        means,
        stds,
        _shapelet_classes,
    ) = transformer.shapelets_

    values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    dilations = np.asarray(dilations, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    normalises = np.asarray(normalises, dtype=bool)
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)

    n_shapelets = int(lengths.shape[0])
    n_channels = int(values.shape[1])

    coef = np.asarray(ridge.coef_, dtype=np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    intercept = np.asarray(ridge.intercept_, dtype=np.float64)
    if intercept.ndim == 0:
        intercept = intercept.reshape(1)

    shapelet_values = []
    for i_shapelet in range(n_shapelets):
        length = int(lengths[i_shapelet])
        shapelet_values.append(values[i_shapelet, :, :length].tolist())

    return {
        "version": version,
        "nShapelets": n_shapelets,
        "nChannels": n_channels,
        "shapelets": {
            "values": shapelet_values,
            "lengths": lengths.astype(int).tolist(),
            "dilations": dilations.astype(int).tolist(),
            "thresholds": thresholds.tolist(),
            "normalise": normalises.astype(bool).tolist(),
            "means": means.tolist(),
            "stds": stds.tolist(),
        },
        "scaler": {
            "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        },
        "classifier": {
            "coef": coef.tolist(),
            "intercept": intercept.tolist(),
            "classes": [str(label) for label in ridge.classes_],
        },
    }


def _extract_default_estimator_parts(estimator: Any) -> tuple[Any, Any]:
    named_steps = getattr(estimator, "named_steps", None)
    if not named_steps:
        raise ValueError(
            "Expected aeon's default StandardScaler -> RidgeClassifierCV pipeline"
        )

    scaler = None
    ridge = None
    for step in named_steps.values():
        if hasattr(step, "mean_") and hasattr(step, "scale_"):
            scaler = step
        if (
            hasattr(step, "coef_")
            and hasattr(step, "intercept_")
            and hasattr(step, "classes_")
        ):
            ridge = step

    if scaler is None or ridge is None:
        raise ValueError(
            "Could not find fitted StandardScaler and RidgeClassifierCV in estimator"
        )
    return scaler, ridge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a pickled fitted aeon RDSTClassifier to JSON"
    )
    parser.add_argument(
        "pickle_path", type=Path, help="Pickle containing fitted classifier"
    )
    parser.add_argument("output_json", type=Path, help="Destination JSON model path")
    args = parser.parse_args()

    with args.pickle_path.open("rb") as fh:
        clf = pickle.load(fh)

    model = export_rdst_classifier(clf)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(model, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
