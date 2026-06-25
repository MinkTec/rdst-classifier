//! Ridge linear classifier inference.
//!
//! Supports binary and multi-class classification, matching the Dart
//! `RidgeClassifier` exactly.
//!
//! ## Binary
//!   - Score:  `s = X[i] · coef[0] + intercept[0]`
//!   - Predict: `s > 0 → classes[1]`, else `classes[0]`
//!   - Proba:   `[1−σ(s), σ(s)]`
//!
//! ## Multi-class
//!   - Scores: `S[r] = X[i] · coef[r] + intercept[r]`  for each class `r`
//!   - Predict: `argmax(S)`
//!   - Proba:   `softmax(S)`

use crate::{
    math::{sigmoid, softmax},
    model::{RidgeParams, ScalerParams},
};

#[derive(Debug, Clone)]
pub(crate) struct PreparedRidgeParams {
    pub coef: Vec<f64>,
    pub intercept: Vec<f64>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl PreparedRidgeParams {
    pub fn new(params: &RidgeParams, scaler: &ScalerParams) -> Self {
        let mut coef = vec![0.0; params.coef.len()];
        let mut intercept = params.intercept.clone();

        for row in 0..params.n_rows {
            let row_off = row * params.n_cols;
            for col in 0..params.n_cols {
                let scale = scaler.scale[col];
                if scale != 0.0 {
                    let adjusted = params.coef[row_off + col] / scale;
                    coef[row_off + col] = adjusted;
                    intercept[row] -= scaler.mean[col] * adjusted;
                }
            }
        }

        Self {
            coef,
            intercept,
            n_rows: params.n_rows,
            n_cols: params.n_cols,
        }
    }
}

/// Predicts a hard class label for each row in `x`.
///
/// `x` has shape `(n_samples, n_features)`, stored row-major.
/// Returns a `Vec<String>` of length `n_samples`.
pub fn predict(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &RidgeParams,
) -> Vec<String> {
    if params.is_binary() {
        (0..n_samples)
            .map(|i| {
                let score =
                    dot_row(x, i, n_features, &params.coef, 0, params.n_cols) + params.intercept[0];
                if score > 0.0 {
                    params.classes[1].clone()
                } else {
                    params.classes[0].clone()
                }
            })
            .collect()
    } else {
        (0..n_samples)
            .map(|i| {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_class = 0usize;
                for r in 0..params.n_rows {
                    let score = dot_row(x, i, n_features, &params.coef, r, params.n_cols)
                        + params.intercept[r];
                    if score > best_score {
                        best_score = score;
                        best_class = r;
                    }
                }
                params.classes[best_class].clone()
            })
            .collect()
    }
}

/// Predicts class probabilities for each row in `x`.
///
/// For binary: `[P(classes[0]), P(classes[1])]` via sigmoid.
/// For multi-class: `softmax(scores)`.
///
/// Returns a `Vec<f64>` of shape `(n_samples, n_classes)`, row-major.
pub fn predict_proba(
    x: &[f64],
    n_samples: usize,
    n_features: usize,
    params: &RidgeParams,
) -> Vec<f64> {
    let n_classes = params.n_classes();
    let mut result = vec![0.0f64; n_samples * n_classes];

    if params.is_binary() {
        for i in 0..n_samples {
            let score =
                dot_row(x, i, n_features, &params.coef, 0, params.n_cols) + params.intercept[0];
            let p = sigmoid(score);
            result[i * 2] = 1.0 - p; // P(classes[0])
            result[i * 2 + 1] = p; // P(classes[1])
        }
    } else {
        let mut scores = vec![0.0f64; params.n_rows];
        for i in 0..n_samples {
            for (r, score) in scores.iter_mut().enumerate().take(params.n_rows) {
                *score =
                    dot_row(x, i, n_features, &params.coef, r, params.n_cols) + params.intercept[r];
            }
            let proba = softmax(&scores);
            for c in 0..n_classes {
                result[i * n_classes + c] = proba[c];
            }
        }
    }
    result
}

pub(crate) fn predict_from_scores(
    scores: &[f64],
    n_samples: usize,
    params: &RidgeParams,
) -> Vec<String> {
    if params.is_binary() {
        (0..n_samples)
            .map(|i| {
                if scores[i] > 0.0 {
                    params.classes[1].clone()
                } else {
                    params.classes[0].clone()
                }
            })
            .collect()
    } else {
        (0..n_samples)
            .map(|i| {
                let row = &scores[i * params.n_rows..(i + 1) * params.n_rows];
                let mut best_score = f64::NEG_INFINITY;
                let mut best_class = 0usize;
                for (class, &score) in row.iter().enumerate() {
                    if score > best_score {
                        best_score = score;
                        best_class = class;
                    }
                }
                params.classes[best_class].clone()
            })
            .collect()
    }
}

pub(crate) fn predict_proba_from_scores(
    scores: &[f64],
    n_samples: usize,
    params: &RidgeParams,
) -> Vec<f64> {
    let n_classes = params.n_classes();
    let mut result = vec![0.0f64; n_samples * n_classes];

    if params.is_binary() {
        for i in 0..n_samples {
            let p = sigmoid(scores[i]);
            result[i * 2] = 1.0 - p;
            result[i * 2 + 1] = p;
        }
    } else {
        for i in 0..n_samples {
            let score_row = &scores[i * params.n_rows..(i + 1) * params.n_rows];
            let proba = softmax(score_row);
            for c in 0..n_classes {
                result[i * n_classes + c] = proba[c];
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Dot product of row `i_x` of `x` (n_cols_x columns) with row `i_coef` of
/// `coef` (n_cols_coef columns). Both rows have the same length `n_cols_x`.
#[inline]
fn dot_row(
    x: &[f64],
    i_x: usize,
    n_cols_x: usize,
    coef: &[f64],
    i_coef: usize,
    n_cols_coef: usize,
) -> f64 {
    let x_row = &x[i_x * n_cols_x..(i_x + 1) * n_cols_x];
    let c_row = &coef[i_coef * n_cols_coef..(i_coef + 1) * n_cols_coef];
    x_row.iter().zip(c_row.iter()).map(|(&a, &b)| a * b).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn binary_params(coef: Vec<f64>, intercept: f64) -> RidgeParams {
        let n = coef.len();
        RidgeParams {
            n_rows: 1,
            n_cols: n,
            coef,
            intercept: vec![intercept],
            classes: vec!["neg".into(), "pos".into()],
        }
    }

    fn multi_params(coef: Vec<Vec<f64>>, intercept: Vec<f64>, classes: Vec<String>) -> RidgeParams {
        let n_rows = coef.len();
        let n_cols = coef[0].len();
        let flat: Vec<f64> = coef.into_iter().flatten().collect();
        RidgeParams {
            n_rows,
            n_cols,
            coef: flat,
            intercept,
            classes,
        }
    }

    // -----------------------------------------------------------------------
    // Binary predict
    // -----------------------------------------------------------------------

    #[test]
    fn binary_positive_score() {
        let p = binary_params(vec![1.0], 0.0);
        assert_eq!(predict(&[2.0], 1, 1, &p), vec!["pos"]);
    }

    #[test]
    fn binary_negative_score() {
        let p = binary_params(vec![1.0], 0.0);
        assert_eq!(predict(&[-1.0], 1, 1, &p), vec!["neg"]);
    }

    #[test]
    fn binary_intercept_shifts_boundary() {
        // score = 1.0*x + 5.0; x=-3 → score=2 > 0 → pos
        let p = binary_params(vec![1.0], 5.0);
        assert_eq!(predict(&[-3.0], 1, 1, &p), vec!["pos"]);
    }

    // -----------------------------------------------------------------------
    // Multiclass predict
    // -----------------------------------------------------------------------

    #[test]
    fn multiclass_argmax() {
        let p = multi_params(
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
            vec![0.0, 0.0, 0.0],
            vec!["a".into(), "b".into(), "c".into()],
        );
        // scores: a=1, b=0, c=0.5 → "a" wins
        assert_eq!(predict(&[1.0, 0.0], 1, 2, &p), vec!["a"]);
    }

    #[test]
    fn multiclass_intercept_breaks_tie() {
        // 3-class model so the multiclass (argmax) branch is used.
        // class 0 intercept=1 → score 1; others 0 → "first" must win.
        let p = multi_params(
            vec![vec![0.0], vec![0.0], vec![0.0]],
            vec![1.0, 0.0, 0.0],
            vec!["first".into(), "second".into(), "third".into()],
        );
        assert_eq!(predict(&[0.0], 1, 1, &p), vec!["first"]);
    }

    // -----------------------------------------------------------------------
    // Binary predict_proba
    // -----------------------------------------------------------------------

    #[test]
    fn binary_proba_sum_one() {
        let p = binary_params(vec![1.0], 0.0);
        let pr = predict_proba(&[2.0], 1, 1, &p);
        assert_abs_diff_eq!(pr[0] + pr[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn binary_proba_positive_score_dominant() {
        let p = binary_params(vec![1.0], 0.0);
        let pr = predict_proba(&[5.0], 1, 1, &p);
        assert!(
            pr[1] > 0.99,
            "P(pos) should be > 0.99 for large positive score"
        );
    }

    #[test]
    fn binary_proba_zero_score_half() {
        let p = binary_params(vec![0.0], 0.0);
        let pr = predict_proba(&[0.0], 1, 1, &p);
        assert_abs_diff_eq!(pr[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(pr[1], 0.5, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // Multiclass predict_proba
    // -----------------------------------------------------------------------

    #[test]
    fn multiclass_proba_sum_one() {
        let p = multi_params(
            vec![vec![1.0], vec![0.0], vec![-1.0]],
            vec![0.0, 0.0, 0.0],
            vec!["a".into(), "b".into(), "c".into()],
        );
        let pr = predict_proba(&[1.0], 1, 1, &p);
        assert_abs_diff_eq!(pr.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn multiclass_proba_highest_score_dominates() {
        let p = multi_params(
            vec![vec![10.0], vec![0.0], vec![-10.0]],
            vec![0.0, 0.0, 0.0],
            vec!["a".into(), "b".into(), "c".into()],
        );
        let pr = predict_proba(&[1.0], 1, 1, &p);
        assert!(pr[0] > 0.99, "class 'a' should dominate");
    }
}
