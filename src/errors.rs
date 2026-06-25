//! Error types for the RDST classifier.

use thiserror::Error;

/// All errors that can occur during model loading or inference.
#[derive(Debug, Error, uniffi::Error)]
pub enum ClassifierError {
    /// The JSON string could not be parsed.
    #[error("JSON parse error: {msg}")]
    JsonParse { msg: String },

    /// A required JSON field was missing or had the wrong type.
    #[error("Invalid model format: {msg}")]
    InvalidModel { msg: String },

    /// The provided input dimensions are inconsistent with the loaded model.
    #[error("Dimension mismatch: {msg}")]
    DimensionMismatch { msg: String },

    /// An I/O error occurred while reading a model archive.
    #[error("I/O error: {msg}")]
    Io { msg: String },
}

impl From<serde_json::Error> for ClassifierError {
    fn from(e: serde_json::Error) -> Self {
        ClassifierError::JsonParse { msg: e.to_string() }
    }
}

impl From<std::io::Error> for ClassifierError {
    fn from(e: std::io::Error) -> Self {
        ClassifierError::Io { msg: e.to_string() }
    }
}
