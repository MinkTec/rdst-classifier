# RDST Classifier (Rust)

This crate provides a pure-Rust inference engine for the
**Random Dilated Shapelet Transform (RDST)** pipeline.

It implements the same model scoring logic used by the workflow described in
Walkling et al.'s paper,
"Wearable Spine Tracker vs. Video-Based Pose Estimation for Human Activity
Recognition" (DOI: [10.3390/s25123806](https://doi.org/10.3390/s25123806)).

Given a serialized RDST model and sensor input, the crate returns hard labels or
probability outputs.

## Features

- `RdstClassifier::from_json` for direct JSON model loading.
- `RdstClassifier::from_tar_gz` for loading a tar-gzipped model bundle.
- `predict` for hard class labels.
- `predict_proba` for per-class probabilities.
- UniFFI object (`FfiRdstClassifier`) for Dart, Swift, and Kotlin integrations.

## Layout

- `model`: parsed model container and parameter blocks.
- `transform`: RDST feature extraction.
- `subsequence` and `math`: subsequence helpers and numeric utilities.
- `classifier`: deterministic scoring.
- `scaler`: standardization of transformed features.
- `model_io`: input format parsing and validation.

## Usage

```rust
use rdst_classifier::RdstClassifier;

let json = std::fs::read_to_string("model.json")?;
let clf = RdstClassifier::from_json(&json)?;

let x = vec![/* flattened samples × channels × timepoints */];
let labels = clf.predict(&x, n_samples, n_channels, n_timepoints)?;
let probas = clf.predict_proba(&x, n_samples, n_channels, n_timepoints)?;
```

## References

- Sensors 25(12):3806, 2025.
- DOI: [10.3390/s25123806](https://doi.org/10.3390/s25123806)

## License

MIT
