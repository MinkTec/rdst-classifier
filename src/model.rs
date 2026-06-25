//! Core data structures for a loaded RDST model.
//!
//! All numeric data is stored in flat `Vec<f64>` arrays with explicit dimension
//! parameters, mirroring the Dart implementation's `Float64List` layout.

/// Parameters for a single dilated shapelet.
///
/// `values` is a 2-D array of shape `(n_channels, length)` stored channel-major:
/// `values[c * length + i]` gives the value at channel `c`, position `i`.
#[derive(Debug, Clone)]
pub struct ShapeletParams {
    /// Flat (channel-major) shapelet values. Length = `n_channels * length`.
    pub values: Vec<f64>,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of time points in the shapelet.
    pub length: usize,
    /// Dilation factor used when extracting subsequences.
    pub dilation: usize,
    /// L1-distance threshold for the occurrence count feature.
    pub threshold: f64,
    /// Whether Z-normalisation is applied to subsequences before matching.
    pub normalise: bool,
    /// Per-channel training mean (stored but not used at inference).
    pub means: Vec<f64>,
    /// Per-channel training std (stored but not used at inference).
    pub stds: Vec<f64>,
}

impl ShapeletParams {
    /// Returns the shapelet value at channel `c`, position `i`.
    #[inline]
    pub fn value_at(&self, c: usize, i: usize) -> f64 {
        self.values[c * self.length + i]
    }
}

/// Parameters for the StandardScaler step.
///
/// Applies: `(x − mean) / scale` element-wise.
#[derive(Debug, Clone)]
pub struct ScalerParams {
    /// Per-feature mean. Length = `3 * n_shapelets`.
    pub mean: Vec<f64>,
    /// Per-feature scale. Length = `3 * n_shapelets`.
    pub scale: Vec<f64>,
}

/// Parameters for the Ridge linear classifier.
///
/// For **binary** classification `coef` has shape `(1, n_features)` and
/// `intercept` has length `1`. Decision rule: `score > 0 → classes[1]`,
/// else `classes[0]`.
///
/// For **multi-class** `coef` has shape `(n_classes, n_features)` and
/// `intercept` has length `n_classes`. Decision rule: argmax of scores.
#[derive(Debug, Clone)]
pub struct RidgeParams {
    /// Coefficient matrix stored row-major: `coef[r * n_cols + c]`.
    /// Shape: `(n_rows, n_cols)` where `n_rows` = 1 (binary) or n_classes.
    pub coef: Vec<f64>,
    /// Number of rows (1 for binary, n_classes for multi-class).
    pub n_rows: usize,
    /// Number of columns (= 3 * n_shapelets).
    pub n_cols: usize,
    /// Bias / intercept. Length = `n_rows`.
    pub intercept: Vec<f64>,
    /// Ordered class labels.
    pub classes: Vec<String>,
}

impl RidgeParams {
    /// Number of output classes.
    #[inline]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// `true` if this is a binary classification problem.
    #[inline]
    pub fn is_binary(&self) -> bool {
        self.classes.len() == 2
    }

    /// Returns `coef[row][col]`.
    #[inline]
    pub fn coef_at(&self, row: usize, col: usize) -> f64 {
        self.coef[row * self.n_cols + col]
    }
}

/// Complete RDST model combining all fitted parameters.
#[derive(Debug, Clone)]
pub struct RdstModel {
    /// Format version string from the JSON file.
    pub version: String,
    /// Total number of shapelets.
    pub n_shapelets: usize,
    /// Number of input channels.
    pub n_channels: usize,
    /// Shapelet parameters, one per shapelet.
    pub shapelets: Vec<ShapeletParams>,
    /// Fitted StandardScaler parameters.
    pub scaler: ScalerParams,
    /// Fitted Ridge classifier parameters.
    pub classifier: RidgeParams,
}
