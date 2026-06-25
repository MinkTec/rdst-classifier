//! Core subsequence extraction and shapelet feature computation.
//!
//! This is the hot-path of the RDST inference pipeline. All operations are
//! implemented as tight loops designed to auto-vectorize to AVX2 (x86-64) or
//! NEON (ARM/Apple Silicon) without platform-specific intrinsics.
//!
//! # Memory layout conventions
//!
//! | Array    | Shape                          | Flat index                               |
//! |----------|--------------------------------|------------------------------------------|
//! | Input X  | (n_channels, n_timepoints)     | `c * n_timepoints + t`                   |
//! | subs     | (n_subs, n_channels, length)   | `s * n_channels * length + c * length + j` |
//! | means    | (n_channels, n_subs)           | `c * n_subs + s`                         |
//! | stds     | (n_channels, n_subs)           | `c * n_subs + s`                         |

/// Threshold below which a standard deviation is treated as zero during
/// Z-normalisation, matching Python's `AEON_NUMBA_STD_THRESHOLD = 1e-8`.
const STD_THRESHOLD: f64 = 1e-8;

// ---------------------------------------------------------------------------
// Subsequence extraction
// ---------------------------------------------------------------------------

/// Returns all dilated subsequences of a single-sample time series.
///
/// `x` has shape `(n_channels, n_timepoints)`, stored channel-major.
/// Returns a `Vec<f64>` of shape `(n_subs, n_channels, length)`, sub-major.
///
/// `n_subs = n_timepoints − (length − 1) * dilation`
pub fn get_all_subsequences(
    x: &[f64],
    n_channels: usize,
    n_timepoints: usize,
    length: usize,
    dilation: usize,
) -> Vec<f64> {
    let n_subs = n_timepoints - (length - 1) * dilation;
    let stride = n_channels * length;
    let mut result = vec![0.0f64; n_subs * stride];

    for i_sub in 0..n_subs {
        let sub_off = i_sub * stride;
        for c in 0..n_channels {
            let x_off = c * n_timepoints;
            let sub_c_off = sub_off + c * length;
            for j in 0..length {
                // SAFETY: bounds guaranteed by n_subs formula
                result[sub_c_off + j] = x[x_off + i_sub + j * dilation];
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Sliding mean and standard deviation
// ---------------------------------------------------------------------------

/// Computes per-channel sliding mean and standard deviation for all dilated
/// subsequences of `x`.
///
/// Uses an incremental (Welford-style) update to avoid O(n·length) recomputation
/// per window, matching the Dart implementation exactly.
///
/// Returns `(means, stds)`, both shaped `(n_channels, n_subs)`, channel-major.
pub fn sliding_mean_std(
    x: &[f64],
    n_channels: usize,
    n_timepoints: usize,
    length: usize,
    dilation: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_subs = n_timepoints - (length - 1) * dilation;

    let mut means = vec![0.0f64; n_channels * n_subs];
    let mut stds = vec![0.0f64; n_channels * n_subs];
    let mut sum = vec![0.0f64; n_channels];
    let mut sum2 = vec![0.0f64; n_channels];

    for i_mod_dil in 0..dilation {
        // Indices of the first subsequence for this congruence class.
        // idx_sub[j] = j * dilation + i_mod_dil
        let last_idx = (length - 1) * dilation + i_mod_dil;
        if last_idx >= n_timepoints {
            continue;
        }

        // Initialise sums for the first subsequence in this congruence class.
        sum.iter_mut().for_each(|v| *v = 0.0);
        sum2.iter_mut().for_each(|v| *v = 0.0);

        for j in 0..length {
            let t = j * dilation + i_mod_dil;
            for c in 0..n_channels {
                let v = x[c * n_timepoints + t];
                sum[c] += v;
                sum2[c] += v * v;
            }
        }

        // Write first subsequence stats.
        write_mean_std(
            (&mut means, &mut stds),
            (&sum, &sum2),
            n_channels,
            n_subs,
            i_mod_dil,
            length,
        );

        // Slide forward in steps of `dilation`.
        let mut i_sub_start = i_mod_dil + dilation;
        let mut first_t = i_mod_dil; // tracks first element index (idxSub[0])
        let mut last_t = last_idx; // tracks last element index  (idxSub[length-1])

        while i_sub_start < n_subs {
            let new_t = last_t + dilation;
            if new_t >= n_timepoints {
                break;
            }
            let old_t = first_t;

            for c in 0..n_channels {
                let v_new = x[c * n_timepoints + new_t];
                let v_old = x[c * n_timepoints + old_t];
                sum[c] += v_new - v_old;
                sum2[c] += v_new * v_new - v_old * v_old;
            }

            write_mean_std(
                (&mut means, &mut stds),
                (&sum, &sum2),
                n_channels,
                n_subs,
                i_sub_start,
                length,
            );

            first_t += dilation;
            last_t += dilation;
            i_sub_start += dilation;
        }
    }

    (means, stds)
}

#[inline(always)]
fn write_mean_std(
    output: (&mut [f64], &mut [f64]),
    sums: (&[f64], &[f64]),
    n_channels: usize,
    n_subs: usize,
    i_sub: usize,
    length: usize,
) {
    let (means, stds) = output;
    let (sum, sum2) = sums;
    let len_f = length as f64;
    for c in 0..n_channels {
        let m = sum[c] / len_f;
        means[c * n_subs + i_sub] = m;
        let variance = sum2[c] / len_f - m * m;
        if variance > STD_THRESHOLD {
            stds[c * n_subs + i_sub] = variance.sqrt();
        }
        // else std stays 0.0 (already initialised)
    }
}

// ---------------------------------------------------------------------------
// Z-normalisation
// ---------------------------------------------------------------------------

/// Z-normalises the subsequences in `subs` using precomputed `means` and `stds`.
///
/// Channels with `std ≤ STD_THRESHOLD` are set to 0.0 (flat channel).
/// Returns a new allocation; `subs` is not modified.
pub fn normalise_subsequences(
    subs: &[f64],
    means: &[f64],
    stds: &[f64],
    n_subs: usize,
    n_channels: usize,
    length: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; subs.len()];

    for i_sub in 0..n_subs {
        for c in 0..n_channels {
            let std = stds[c * n_subs + i_sub];
            if std > STD_THRESHOLD {
                let mean = means[c * n_subs + i_sub];
                let src_off = i_sub * n_channels * length + c * length;
                let dst_off = src_off;
                for j in 0..length {
                    result[dst_off + j] = (subs[src_off + j] - mean) / std;
                }
            }
            // else: stays 0.0 — correct normalisation when std ≈ 0
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Shapelet features (hot path)
// ---------------------------------------------------------------------------

/// Computes the three RDST features `(min_dist, arg_min, occurrence)` for a
/// single shapelet applied to all `n_subs` subsequences.
///
/// Distance metric: **L1 (Manhattan)**, summed over all channels × positions.
/// Returns `(min_dist, arg_min_as_f64, occurrence_count_as_f64)`.
///
/// This loop is intentionally structured to hint LLVM towards auto-vectorisation
/// of the inner distance accumulation.
pub fn compute_shapelet_features(
    subs: &[f64],
    shp_values: &[f64],
    threshold: f64,
    n_subs: usize,
    n_channels: usize,
    length: usize,
) -> (f64, f64, f64) {
    let stride = n_channels * length;

    let mut min_dist = f64::INFINITY;
    let mut arg_min = 0usize;
    let mut occurrence = 0.0f64;

    for i_sub in 0..n_subs {
        // L1 distance between subsequence i_sub and the shapelet.
        // The inner slice has exactly `stride` elements; written as a
        // single iterator chain so LLVM can vectorise it.
        let sub_slice = &subs[i_sub * stride..(i_sub + 1) * stride];
        let dist: f64 = sub_slice
            .iter()
            .zip(shp_values.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();

        if dist < min_dist {
            min_dist = dist;
            arg_min = i_sub;
        }
        if dist < threshold {
            occurrence += 1.0;
        }
    }

    (min_dist, arg_min as f64, occurrence)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // -----------------------------------------------------------------------
    // get_all_subsequences
    // -----------------------------------------------------------------------

    #[test]
    fn subs_single_channel_dilation1() {
        // X = [1,2,3,4,5], single channel, length=3, dilation=1 → n_subs=3
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let subs = get_all_subsequences(&x, 1, 5, 3, 1);
        assert_eq!(subs.len(), 9);
        // sub0: [1,2,3]  sub1: [2,3,4]  sub2: [3,4,5]
        assert_eq!(&subs[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&subs[3..6], &[2.0, 3.0, 4.0]);
        assert_eq!(&subs[6..9], &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn subs_single_channel_dilation2() {
        // X = [1,2,3,4,5], length=2, dilation=2 → n_subs = 5 - 1*2 = 3
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let subs = get_all_subsequences(&x, 1, 5, 2, 2);
        // sub0: [x[0], x[2]] = [1,3]
        // sub1: [x[1], x[3]] = [2,4]
        // sub2: [x[2], x[4]] = [3,5]
        assert_eq!(&subs[0..2], &[1.0, 3.0]);
        assert_eq!(&subs[2..4], &[2.0, 4.0]);
        assert_eq!(&subs[4..6], &[3.0, 5.0]);
    }

    #[test]
    fn subs_multi_channel_layout() {
        // X = [[1,2,3],[4,5,6]], 2 channels, 3 timepoints, length=2, dilation=1 → n_subs=2
        // channel-major: x = [1,2,3, 4,5,6]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let subs = get_all_subsequences(&x, 2, 3, 2, 1);
        // sub0: [ch0[0..2], ch1[0..2]] = [1,2, 4,5]
        // sub1: [ch0[1..3], ch1[1..3]] = [2,3, 5,6]
        assert_eq!(&subs[0..4], &[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(&subs[4..8], &[2.0, 3.0, 5.0, 6.0]);
    }

    // -----------------------------------------------------------------------
    // sliding_mean_std
    // -----------------------------------------------------------------------

    #[test]
    fn sliding_constant_series_std_zero() {
        let x = vec![3.0; 5]; // constant
        let (means, stds) = sliding_mean_std(&x, 1, 5, 3, 1);
        assert_eq!(means.len(), 3);
        for m in &means {
            assert_abs_diff_eq!(*m, 3.0, epsilon = 1e-12);
        }
        for s in &stds {
            assert_abs_diff_eq!(*s, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn sliding_known_mean() {
        // X = [1,2,3,4,5], length=3, dilation=1 → n_subs=3
        // means: [2, 3, 4]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (means, _) = sliding_mean_std(&x, 1, 5, 3, 1);
        assert_abs_diff_eq!(means[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(means[1], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(means[2], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn sliding_known_std() {
        // X = [1,2,3], length=3, dilation=1 → n_subs=1
        // var = E[x²] - E[x]² = (1+4+9)/3 - 4 = 14/3 - 4 = 2/3
        // std = sqrt(2/3) ≈ 0.8165
        let x = vec![1.0, 2.0, 3.0];
        let (_, stds) = sliding_mean_std(&x, 1, 3, 3, 1);
        let expected = (2.0f64 / 3.0).sqrt();
        assert_abs_diff_eq!(stds[0], expected, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // normalise_subsequences
    // -----------------------------------------------------------------------

    #[test]
    fn normalise_zero_mean_unit_var() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let subs = get_all_subsequences(&x, 1, 5, 5, 1); // single sub = entire series
        let (means, stds) = sliding_mean_std(&x, 1, 5, 5, 1);
        let norm = normalise_subsequences(&subs, &means, &stds, 1, 1, 5);
        let mean: f64 = norm.iter().sum::<f64>() / 5.0;
        let var: f64 = norm.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / 5.0;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(var.sqrt(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn normalise_constant_channel_zeroed() {
        // Constant channel should produce all-zeros after normalisation.
        let x = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let subs = get_all_subsequences(&x, 1, 5, 3, 1);
        let (means, stds) = sliding_mean_std(&x, 1, 5, 3, 1);
        let norm = normalise_subsequences(&subs, &means, &stds, 3, 1, 3);
        for v in &norm {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-12);
        }
    }

    // -----------------------------------------------------------------------
    // compute_shapelet_features
    // -----------------------------------------------------------------------

    #[test]
    fn shapelet_exact_match() {
        // shapelet = [1,2,3]; subs = [[0,0,0],[1,2,3],[2,4,6]]
        // dist to sub0 = 1+2+3=6, sub1 = 0, sub2 = 1+2+3=6
        let subs = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
        let shp = vec![1.0, 2.0, 3.0];
        let (min_dist, arg_min, _) = compute_shapelet_features(&subs, &shp, 100.0, 3, 1, 3);
        assert_abs_diff_eq!(min_dist, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(arg_min, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn shapelet_occurrence_count() {
        // threshold=2: only sub1 (dist=0) qualifies
        let subs = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
        let shp = vec![1.0, 2.0, 3.0];
        let (_, _, occ) = compute_shapelet_features(&subs, &shp, 2.0, 3, 1, 3);
        assert_abs_diff_eq!(occ, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn shapelet_occurrence_strict_less() {
        // threshold=0: no subsequence qualifies (strict <, not <=)
        let subs = vec![1.0, 2.0, 3.0]; // single sub, exact match
        let shp = vec![1.0, 2.0, 3.0];
        let (_, _, occ) = compute_shapelet_features(&subs, &shp, 0.0, 1, 1, 3);
        assert_abs_diff_eq!(occ, 0.0, epsilon = 1e-12);
    }
}
