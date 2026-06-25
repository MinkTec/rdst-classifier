//! Math utility functions used throughout the RDST inference pipeline.

/// Computes the numerically stable softmax of a slice, returning a new `Vec<f64>`.
///
/// Subtracts `max(scores)` before exponentiation to avoid overflow.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = out.iter().sum();
    out.iter_mut().for_each(|v| *v /= sum);
    out
}

/// Computes the sigmoid of a scalar value: `1 / (1 + exp(-x))`.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Returns the `p`-th percentile (0–100) of `values` using linear interpolation,
/// matching NumPy's default behaviour.
///
/// `values` is sorted internally; the original slice is not modified.
///
/// # Panics
/// Panics if `values` is empty.
pub fn percentile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty(), "percentile: values must not be empty");
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let idx = (p / 100.0) * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let frac = idx - lower as f64;
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn softmax_sums_to_one() {
        let v = softmax(&[1.0, 2.0, 3.0]);
        assert_abs_diff_eq!(v.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn softmax_largest_wins() {
        let v = softmax(&[0.0, 10.0, 0.0]);
        assert!(v[1] > v[0] && v[1] > v[2]);
    }

    #[test]
    fn softmax_uniform() {
        let v = softmax(&[1.0, 1.0, 1.0]);
        for &x in &v {
            assert_abs_diff_eq!(x, 1.0 / 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn softmax_numerically_stable() {
        let v = softmax(&[1000.0, 1001.0, 999.0]);
        assert_abs_diff_eq!(v.iter().sum::<f64>(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn softmax_single() {
        let v = softmax(&[42.0]);
        assert_abs_diff_eq!(v[0], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn sigmoid_zero() {
        assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn sigmoid_large_positive() {
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn sigmoid_large_negative() {
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn sigmoid_symmetry() {
        let x = 2.5;
        assert_abs_diff_eq!(sigmoid(-x), 1.0 - sigmoid(x), epsilon = 1e-15);
    }

    #[test]
    fn percentile_endpoints() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(percentile(&v, 0.0), 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(percentile(&v, 100.0), 5.0, epsilon = 1e-15);
    }

    #[test]
    fn percentile_median_odd() {
        let v = vec![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(percentile(&v, 50.0), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn percentile_interpolation() {
        let v = vec![0.0, 1.0, 2.0, 3.0];
        // 25th percentile of [0,1,2,3]: idx = 0.25*3 = 0.75 → 0.0*0.25 + 1.0*0.75 = 0.75
        assert_abs_diff_eq!(percentile(&v, 25.0), 0.75, epsilon = 1e-12);
    }
}
