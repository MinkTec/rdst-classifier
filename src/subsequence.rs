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
    get_all_subsequences_into(&mut result, x, n_channels, n_timepoints, length, dilation);
    result
}

/// Writes all dilated subsequences of a single-sample time series into `result`.
pub fn get_all_subsequences_into(
    result: &mut Vec<f64>,
    x: &[f64],
    n_channels: usize,
    n_timepoints: usize,
    length: usize,
    dilation: usize,
) {
    let n_subs = n_timepoints - (length - 1) * dilation;
    let stride = n_channels * length;
    result.resize(n_subs * stride, 0.0);

    if dilation == 1 {
        for i_sub in 0..n_subs {
            let sub_off = i_sub * stride;
            for c in 0..n_channels {
                let x_start = c * n_timepoints + i_sub;
                let sub_c_off = sub_off + c * length;
                result[sub_c_off..sub_c_off + length]
                    .copy_from_slice(&x[x_start..x_start + length]);
            }
        }
        return;
    }

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
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut sum = Vec::new();
    let mut sum2 = Vec::new();
    sliding_mean_std_into(
        &mut means,
        &mut stds,
        &mut sum,
        &mut sum2,
        x,
        n_channels,
        n_timepoints,
        length,
        dilation,
    );
    (means, stds)
}

/// Computes per-channel sliding mean/std into reusable output and scratch buffers.
pub fn sliding_mean_std_into(
    means: &mut Vec<f64>,
    stds: &mut Vec<f64>,
    sum: &mut Vec<f64>,
    sum2: &mut Vec<f64>,
    x: &[f64],
    n_channels: usize,
    n_timepoints: usize,
    length: usize,
    dilation: usize,
) {
    let n_subs = n_timepoints - (length - 1) * dilation;

    means.resize(n_channels * n_subs, 0.0);
    stds.resize(n_channels * n_subs, 0.0);
    sum.resize(n_channels, 0.0);
    sum2.resize(n_channels, 0.0);

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
            (means.as_mut_slice(), stds.as_mut_slice()),
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
                (means.as_mut_slice(), stds.as_mut_slice()),
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
        } else {
            stds[c * n_subs + i_sub] = 0.0;
        }
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
    let mut result = subs.to_vec();
    normalise_subsequences_in_place(&mut result, means, stds, n_subs, n_channels, length);
    result
}

/// Z-normalises subsequences in place using precomputed `means` and `stds`.
pub fn normalise_subsequences_in_place(
    subs: &mut [f64],
    means: &[f64],
    stds: &[f64],
    n_subs: usize,
    n_channels: usize,
    length: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if length >= 8 && std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                normalise_subsequences_in_place_avx2(subs, means, stds, n_subs, n_channels, length);
            }
            return;
        }
    }

    normalise_subsequences_in_place_scalar(subs, means, stds, n_subs, n_channels, length);
}

fn normalise_subsequences_in_place_scalar(
    subs: &mut [f64],
    means: &[f64],
    stds: &[f64],
    n_subs: usize,
    n_channels: usize,
    length: usize,
) {
    let stride = n_channels * length;

    for i_sub in 0..n_subs {
        for c in 0..n_channels {
            let std = stds[c * n_subs + i_sub];
            let off = i_sub * stride + c * length;
            if std > STD_THRESHOLD {
                let mean = means[c * n_subs + i_sub];
                for j in 0..length {
                    subs[off + j] = (subs[off + j] - mean) / std;
                }
            } else {
                subs[off..off + length].fill(0.0);
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn normalise_subsequences_in_place_avx2(
    subs: &mut [f64],
    means: &[f64],
    stds: &[f64],
    n_subs: usize,
    n_channels: usize,
    length: usize,
) {
    let stride = n_channels * length;

    for i_sub in 0..n_subs {
        for c in 0..n_channels {
            let mean_std_off = c * n_subs + i_sub;
            let std = stds[mean_std_off];
            let off = i_sub * stride + c * length;

            if std > STD_THRESHOLD {
                let mean_v = arch::_mm256_set1_pd(means[mean_std_off]);
                let inv_std_v = arch::_mm256_set1_pd(1.0 / std);
                let mut j = 0usize;

                while j + 4 <= length {
                    let ptr = unsafe { subs.as_mut_ptr().add(off + j) };
                    let values = unsafe { arch::_mm256_loadu_pd(ptr) };
                    let centered = arch::_mm256_sub_pd(values, mean_v);
                    let normalised = arch::_mm256_mul_pd(centered, inv_std_v);
                    unsafe { arch::_mm256_storeu_pd(ptr, normalised) };
                    j += 4;
                }

                while j < length {
                    let idx = off + j;
                    subs[idx] = (subs[idx] - means[mean_std_off]) / std;
                    j += 1;
                }
            } else {
                subs[off..off + length].fill(0.0);
            }
        }
    }
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if stride >= 4 && std::arch::is_x86_feature_detected!("avx2") {
            return unsafe {
                compute_shapelet_features_avx2(subs, shp_values, threshold, n_subs, stride)
            };
        }
    }

    compute_shapelet_features_scalar(subs, shp_values, threshold, n_subs, stride)
}

fn compute_shapelet_features_scalar(
    subs: &[f64],
    shp_values: &[f64],
    threshold: f64,
    n_subs: usize,
    stride: usize,
) -> (f64, f64, f64) {
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

#[cfg(target_arch = "x86")]
use std::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64 as arch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn compute_shapelet_features_avx2(
    subs: &[f64],
    shp_values: &[f64],
    threshold: f64,
    n_subs: usize,
    stride: usize,
) -> (f64, f64, f64) {
    let mut min_dist = f64::INFINITY;
    let mut arg_min = 0usize;
    let mut occurrence = 0.0f64;

    for i_sub in 0..n_subs {
        let sub_slice = &subs[i_sub * stride..(i_sub + 1) * stride];
        let dist = unsafe { l1_distance_avx2(sub_slice, shp_values) };

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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn l1_distance_avx2(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let mut i = 0usize;
    let len = a.len();
    let mut acc = arch::_mm256_setzero_pd();
    let abs_mask = arch::_mm256_castsi256_pd(arch::_mm256_set1_epi64x(0x7fff_ffff_ffff_ffff));

    while i + 4 <= len {
        let av = unsafe { arch::_mm256_loadu_pd(a.as_ptr().add(i)) };
        let bv = unsafe { arch::_mm256_loadu_pd(b.as_ptr().add(i)) };
        let diff = arch::_mm256_sub_pd(av, bv);
        let abs = arch::_mm256_and_pd(diff, abs_mask);
        acc = arch::_mm256_add_pd(acc, abs);
        i += 4;
    }

    let mut lanes = [0.0f64; 4];
    unsafe { arch::_mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f64>();

    while i < len {
        sum += (unsafe { *a.get_unchecked(i) } - unsafe { *b.get_unchecked(i) }).abs();
        i += 1;
    }

    sum
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
