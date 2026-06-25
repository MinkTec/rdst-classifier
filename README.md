# RDST Classifier (Rust)

This crate provides a pure-Rust inference engine for trained
**Random Dilated Shapelet Transform (RDST)** time-series classifiers.

The usual workflow is:

1. Train an RDST model in Python with [aeon](https://www.aeon-toolkit.org/).
2. Export the fitted model to JSON with `scripts/aeon_rdst_export.py`.
3. Load that JSON in Rust and classify numeric time-series data.

The library is domain-agnostic. It can be used for wearable sensors, industrial
signals, medical measurements, motion capture, audio-derived features, or any
other fixed-length univariate or multivariate time-series data that can be
represented as floating-point numbers.

The `paper-flextail-vs-camera` repository is treated as an example use case and
regression fixture, not as a requirement for the crate.

## For Data Scientists

You can do the model-development work in Python. Use aeon to train and evaluate
the classifier, export `model.json`, and hand that file plus your numeric input
array to the Rust side of your application. The Rust API does not require you to
understand ownership, lifetimes, or other advanced Rust concepts; it loads the
model and scores a flat list of `float64` values.

## Features

- `RdstClassifier::from_json` for direct JSON model loading.
- `RdstClassifier::from_tar_gz` for loading a tar-gzipped model bundle.
- `predict` for hard class labels matching aeon's default RDST predictions.
- `predict_proba` for score-based per-class probabilities.
- UniFFI object (`FfiRdstClassifier`) for Dart, Swift, and Kotlin integrations.

## Data Shape

RDST expects data with shape:

```text
n_samples x n_channels x n_timepoints
```

Rust receives this as one flat `Vec<f64>` in this order:

```text
sample 0, channel 0, all timepoints
sample 0, channel 1, all timepoints
...
sample 1, channel 0, all timepoints
```

If your Python data is a NumPy array shaped `samples x timepoints x channels`,
transpose it before exporting or sending it to Rust:

```python
X_rdst = X.transpose(0, 2, 1)  # samples x channels x timepoints
flat = X_rdst.astype("float64").ravel().tolist()
```

All samples passed in one call must have the same channel count and time length.
If your recordings have different lengths, window or pad them before training
and inference.

## Layout

- `model`: parsed model container and parameter blocks.
- `transform`: RDST feature extraction.
- `subsequence` and `math`: subsequence helpers and numeric utilities.
- `classifier`: deterministic scoring.
- `scaler`: standardization of transformed features.
- `model_io`: input format parsing and validation.

## Usage

Minimal Rust inference code:

```rust
use rdst_classifier::RdstClassifier;

let json = std::fs::read_to_string("model.json")?;
let clf = RdstClassifier::from_json(&json)?;

let x = vec![/* flattened samples × channels × timepoints */];
let labels = clf.predict(&x, n_samples, n_channels, n_timepoints)?;
let probas = clf.predict_proba(&x, n_samples, n_channels, n_timepoints)?;
```

If you are new to Rust, the important part is that `x` is just a flat list of
`f64` numbers and the three dimension arguments tell Rust how to interpret it.
The model JSON is produced by Python; you do not need to train the model in Rust.

## Training And Exporting From Aeon

This crate is inference-only. Train the RDST model with aeon, then export the
fitted shapelets, scaler, and ridge classifier to JSON:

```python
import json
from pathlib import Path

from aeon.classification.shapelet_based import RDSTClassifier

from scripts.aeon_rdst_export import export_rdst_classifier

clf = RDSTClassifier(max_shapelets=10_000, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)  # X shape: n_cases x n_channels x n_timepoints

model = export_rdst_classifier(clf)
Path("model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
```

The exporter supports aeon's default RDST pipeline:
`RandomDilatedShapeletTransform -> StandardScaler -> RidgeClassifierCV`.
It writes the exact fields consumed by `RdstClassifier::from_json`.

Parity with aeon is defined for hard-label `predict` outputs. Aeon's default
`RDSTClassifier.predict_proba` falls back to one-hot predictions because
`RidgeClassifierCV` has no probability API; this crate's `predict_proba` keeps
the existing score-based sigmoid/softmax behavior.

## Paper Use Case Fixture

The `paper-flextail-vs-camera` repository is used as one concrete example of
real multivariate sensor data. It is not a submodule, is not part of the public
model format, and is not required for other datasets.

To regenerate the aeon equivalence fixture, run the example script. If the paper
repo is not already available, the script clones it into
`target/example-data/paper-flextail-vs-camera`:

```sh
uv run --with aeon --with scikit-learn --with numpy python scripts/generate_aeon_paper_fixture.py
cargo test predictions_match_aeon_on_paper_rsf_fixture
```

That fixture trains a small aeon RDST model on windows extracted from the paper
repo's `.rsf` recordings, exports it to JSON, and checks that Rust returns the
same hard-label predictions on held-out windows.

## References

- [aeon: a toolkit for time-series machine learning](https://www.aeon-toolkit.org/)
- Middlehurst, Schafer, and Bagnall. [Bake off redux: a review and experimental evaluation of recent time series classification algorithms](https://arxiv.org/abs/2304.13029), arXiv:2304.13029.
- Walkling et al. "Wearable Spine Tracker vs. Video-Based Pose Estimation for Human Activity Recognition", Sensors 25(12):3806, 2025. DOI: [10.3390/s25123806](https://doi.org/10.3390/s25123806)

## License

Apache-2.0
