//! Pure-Rust inference engine for trained **Random Dilated Shapelet Transform
//! (RDST)** time-series classifiers.
//!
//! Train an RDST model in Python with aeon, export the fitted shapelets, scaler,
//! and ridge classifier to JSON, then use this crate to run deterministic
//! inference from Rust or through the UniFFI bindings. The implementation is
//! generic over numeric time-series data; the input only needs to be shaped as
//! `n_samples × n_channels × n_timepoints`.
//!
//! # Pipeline
//!
//! ```text
//! Raw time series  (f64 slice, shape: n_samples × n_channels × n_timepoints)
//!         │
//!         ▼
//! 1. RdstTransform  — dilated shapelet feature extraction → (n_samples × 3·n_shapelets)
//!         │
//!         ▼
//! 2. StandardScaler — (x − mean) / scale, element-wise
//!         │
//!         ▼
//! 3. RidgeClassifier — linear dot product → hard label or probability vector
//! ```
//!
//! # Quick start
//!
//! ```ignore
//! use rdst_classifier::RdstClassifier;
//!
//! let json = std::fs::read_to_string("model.json").unwrap();
//! let clf = RdstClassifier::from_json(&json).unwrap();
//!
//! // Input: flat slice, layout X[s, c, t] = X[s*nC*nT + c*nT + t]
//! let labels = clf.predict(&input, n_samples, n_channels, n_timepoints).unwrap();
//! let probas = clf.predict_proba(&input, n_samples, n_channels, n_timepoints).unwrap();
//! ```
//!
//! # UniFFI bindings
//!
//! The [`FfiRdstClassifier`] type exposes the same API via UniFFI for use from
//! Dart, Swift, and Kotlin.

uniffi::setup_scaffolding!();

pub mod classifier;
pub mod errors;
pub mod math;
pub mod model;
pub mod model_io;
pub mod scaler;
pub mod subsequence;
pub mod transform;

pub use errors::ClassifierError;
pub use model::{RdstModel, RidgeParams, ScalerParams, ShapeletParams};

use std::io::Read;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Pure-Rust high-level API
// ---------------------------------------------------------------------------

/// High-level inference pipeline for a trained RDST classifier.
///
/// Loads a trained model and runs:
/// `RdstTransform → StandardScaler → RidgeClassifier`
///
/// # Example
///
/// ```ignore
/// use rdst_classifier::RdstClassifier;
///
/// let json = std::fs::read_to_string("model.json").unwrap();
/// let clf = RdstClassifier::from_json(&json).unwrap();
///
/// // Input: flat slice, layout X[s, c, t] = X[s*nC*nT + c*nT + t]
/// let labels = clf.predict(&input, n_samples, n_channels, n_timepoints).unwrap();
/// let probas = clf.predict_proba(&input, n_samples, n_channels, n_timepoints).unwrap();
/// ```
pub struct RdstClassifier {
    model: RdstModel,
}

impl RdstClassifier {
    /// Loads a model from a JSON string produced by the aeon export helper.
    pub fn from_json(json: &str) -> Result<Self, ClassifierError> {
        let model = model_io::from_json(json)?;
        Ok(Self { model })
    }

    /// Loads a model from a `.tar.gz` archive containing a single JSON file.
    pub fn from_tar_gz(bytes: &[u8]) -> Result<Self, ClassifierError> {
        let json = extract_json_from_tar_gz(bytes)?;
        Self::from_json(&json)
    }

    /// Returns the ordered list of class labels this model was trained on.
    pub fn classes(&self) -> &[String] {
        &self.model.classifier.classes
    }

    /// Returns a reference to the underlying [`RdstModel`].
    pub fn model(&self) -> &RdstModel {
        &self.model
    }

    /// Predicts a hard class label for each sample in `x`.
    ///
    /// `x` is a flat slice of shape `(n_samples, n_channels, n_timepoints)`:
    /// `x[s * n_channels * n_timepoints + c * n_timepoints + t]`
    ///
    /// Returns a `Vec<String>` of length `n_samples`.
    pub fn predict(
        &self,
        x: &[f64],
        n_samples: usize,
        n_channels: usize,
        n_timepoints: usize,
    ) -> Result<Vec<String>, ClassifierError> {
        self.check_dims(x, n_samples, n_channels, n_timepoints)?;
        let features = self.extract_features(x, n_samples, n_channels, n_timepoints);
        let n_features = self.model.n_shapelets * 3;
        Ok(classifier::predict(
            &features,
            n_samples,
            n_features,
            &self.model.classifier,
        ))
    }

    /// Predicts class probabilities for each sample in `x`.
    ///
    /// Returns a `Vec<f64>` of shape `(n_samples, n_classes)`, row-major.
    /// Binary: sigmoid-derived; multi-class: softmax-derived.
    pub fn predict_proba(
        &self,
        x: &[f64],
        n_samples: usize,
        n_channels: usize,
        n_timepoints: usize,
    ) -> Result<Vec<f64>, ClassifierError> {
        self.check_dims(x, n_samples, n_channels, n_timepoints)?;
        let features = self.extract_features(x, n_samples, n_channels, n_timepoints);
        let n_features = self.model.n_shapelets * 3;
        Ok(classifier::predict_proba(
            &features,
            n_samples,
            n_features,
            &self.model.classifier,
        ))
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn extract_features(
        &self,
        x: &[f64],
        n_samples: usize,
        n_channels: usize,
        n_timepoints: usize,
    ) -> Vec<f64> {
        let n_features = self.model.n_shapelets * 3;
        let raw = transform::transform(x, n_samples, n_channels, n_timepoints, &self.model);
        scaler::scale(&raw, n_samples, n_features, &self.model.scaler)
    }

    fn check_dims(
        &self,
        x: &[f64],
        n_samples: usize,
        n_channels: usize,
        n_timepoints: usize,
    ) -> Result<(), ClassifierError> {
        let expected = n_samples * n_channels * n_timepoints;
        if x.len() != expected {
            return Err(ClassifierError::DimensionMismatch {
                msg: format!(
                    "x has {} elements but n_samples={} × n_channels={} × n_timepoints={} = {}",
                    x.len(),
                    n_samples,
                    n_channels,
                    n_timepoints,
                    expected
                ),
            });
        }
        if n_channels != self.model.n_channels {
            return Err(ClassifierError::DimensionMismatch {
                msg: format!(
                    "model expects {} channels but input has {}",
                    self.model.n_channels, n_channels
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UniFFI-exposed type
// ---------------------------------------------------------------------------

/// UniFFI-compatible wrapper around [`RdstClassifier`].
///
/// Exposes the same inference API via UniFFI for Dart, Swift, and Kotlin.
/// Dimensions are expressed as `u32` to satisfy UniFFI's type requirements.
#[derive(uniffi::Object)]
pub struct FfiRdstClassifier {
    inner: RdstClassifier,
}

#[uniffi::export]
impl FfiRdstClassifier {
    /// Loads a model from a JSON string.
    #[uniffi::constructor]
    pub fn from_json(json: String) -> Result<Arc<Self>, ClassifierError> {
        let inner = RdstClassifier::from_json(&json)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Loads a model from the raw bytes of a `.tar.gz` archive.
    #[uniffi::constructor]
    pub fn from_tar_gz(bytes: Vec<u8>) -> Result<Arc<Self>, ClassifierError> {
        let inner = RdstClassifier::from_tar_gz(&bytes)?;
        Ok(Arc::new(Self { inner }))
    }

    /// Returns the ordered class labels.
    pub fn classes(&self) -> Vec<String> {
        self.inner.classes().to_vec()
    }

    /// Predicts hard class labels.
    ///
    /// `x` must be a flat `Vec<f64>` of length
    /// `n_samples × n_channels × n_timepoints`.
    pub fn predict(
        &self,
        x: Vec<f64>,
        n_samples: u32,
        n_channels: u32,
        n_timepoints: u32,
    ) -> Result<Vec<String>, ClassifierError> {
        self.inner.predict(
            &x,
            n_samples as usize,
            n_channels as usize,
            n_timepoints as usize,
        )
    }

    /// Predicts class probabilities.
    ///
    /// Returns a flat `Vec<f64>` of shape `n_samples × n_classes`, row-major.
    pub fn predict_proba(
        &self,
        x: Vec<f64>,
        n_samples: u32,
        n_channels: u32,
        n_timepoints: u32,
    ) -> Result<Vec<f64>, ClassifierError> {
        self.inner.predict_proba(
            &x,
            n_samples as usize,
            n_channels as usize,
            n_timepoints as usize,
        )
    }
}

// ---------------------------------------------------------------------------
// .tar.gz helper
// ---------------------------------------------------------------------------

/// Extracts the first JSON file from a `.tar.gz` byte buffer.
fn extract_json_from_tar_gz(bytes: &[u8]) -> Result<String, ClassifierError> {
    let gz = flate2::read::GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(gz);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let mut contents = String::new();
            entry.read_to_string(&mut contents)?;
            return Ok(contents);
        }
    }

    Err(ClassifierError::InvalidModel {
        msg: "no JSON file found in archive".to_owned(),
    })
}
