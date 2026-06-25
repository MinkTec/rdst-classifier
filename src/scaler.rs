//! StandardScaler: applies `(x − mean) / scale` element-wise.

use crate::model::ScalerParams;

/// Scales a feature matrix `x` (shape `n_samples × n_features`) using fitted
/// `params`. Returns a new allocation; `x` is not modified.
///
/// When `scale[j] == 0`, the corresponding output column is set to 0.0
/// (avoids NaN).
pub fn scale(x: &[f64], n_samples: usize, n_features: usize, params: &ScalerParams) -> Vec<f64> {
    let mut result = x.to_vec();
    scale_in_place(&mut result, n_samples, n_features, params);
    result
}

/// Scales a feature matrix in place.
pub fn scale_in_place(x: &mut [f64], n_samples: usize, n_features: usize, params: &ScalerParams) {
    debug_assert_eq!(x.len(), n_samples * n_features);
    for row in x.chunks_mut(n_features).take(n_samples) {
        for (j, value) in row.iter_mut().enumerate() {
            let s = params.scale[j];
            *value = if s == 0.0 {
                0.0
            } else {
                (*value - params.mean[j]) / s
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn params(mean: Vec<f64>, scale: Vec<f64>) -> ScalerParams {
        ScalerParams { mean, scale }
    }

    #[test]
    fn identity_transform() {
        let p = params(vec![0.0, 0.0], vec![1.0, 1.0]);
        let x = vec![3.0, 4.0];
        let out = scale(&x, 1, 2, &p);
        assert_abs_diff_eq!(out[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn correct_scaling() {
        let p = params(vec![1.0, 2.0], vec![2.0, 4.0]);
        let x = vec![3.0, 6.0];
        let out = scale(&x, 1, 2, &p);
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-12); // (3-1)/2
        assert_abs_diff_eq!(out[1], 1.0, epsilon = 1e-12); // (6-2)/4
    }

    #[test]
    fn zero_scale_outputs_zero() {
        let p = params(vec![0.0, 0.0], vec![0.0, 1.0]);
        let x = vec![99.0, 99.0];
        let out = scale(&x, 1, 2, &p);
        assert_abs_diff_eq!(out[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], 99.0, epsilon = 1e-12);
    }

    #[test]
    fn does_not_mutate_input() {
        let p = params(vec![1.0], vec![2.0]);
        let x = vec![5.0];
        let x_copy = x.clone();
        let _ = scale(&x, 1, 1, &p);
        assert_eq!(x, x_copy);
    }

    #[test]
    fn multi_sample_per_feature_independence() {
        let p = params(vec![0.0, 0.0], vec![2.0, 4.0]);
        let x = vec![2.0, 8.0, 4.0, 12.0]; // sample0=[2,8], sample1=[4,12]
        let out = scale(&x, 2, 2, &p);
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-12); // 2/2
        assert_abs_diff_eq!(out[1], 2.0, epsilon = 1e-12); // 8/4
        assert_abs_diff_eq!(out[2], 2.0, epsilon = 1e-12); // 4/2
        assert_abs_diff_eq!(out[3], 3.0, epsilon = 1e-12); // 12/4
    }
}
