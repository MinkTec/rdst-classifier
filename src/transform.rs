//! RDST feature-extraction transform.
//!
//! Given a batch of time-series samples, produces a feature matrix where each
//! sample is described by `3 * n_shapelets` scalars:
//!   `[min_dist, arg_min, occurrence]` per shapelet.
//!
//! Samples are processed in parallel via Rayon; subsequences are computed only
//! once per unique `(length, dilation)` pair per sample (shared computation).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{
    classifier::PreparedRidgeParams,
    model::RdstModel,
    subsequence::{
        compute_shapelet_features, get_all_subsequences_into, normalise_subsequences_in_place,
        sliding_mean_std_into,
    },
};

/// Transforms a batch of time-series samples into shapelet features.
///
/// # Arguments
///
/// * `x` – Flat `&[f64]` of shape `(n_samples, n_channels, n_timepoints)`.
///   Layout: `x[s * n_channels * n_timepoints + c * n_timepoints + t]`.
/// * `n_samples`, `n_channels`, `n_timepoints` – dimensions of `x`.
/// * `model` – Loaded RDST model.
///
/// # Returns
///
/// `Vec<f64>` of shape `(n_samples, 3 * n_shapelets)`, row-major.
pub fn transform(
    x: &[f64],
    n_samples: usize,
    n_channels: usize,
    n_timepoints: usize,
    model: &RdstModel,
) -> Vec<f64> {
    // Identify unique (length, dilation) pairs and which shapelet indices
    // belong to each pair — computed once, shared across all samples.
    let groups = build_groups(model);
    transform_with_groups(x, n_samples, n_channels, n_timepoints, model, &groups)
}

pub(crate) fn transform_with_groups(
    x: &[f64],
    n_samples: usize,
    n_channels: usize,
    n_timepoints: usize,
    model: &RdstModel,
    groups: &[ShapeletGroup],
) -> Vec<f64> {
    let n_shapelets = model.n_shapelets;
    let n_features = 3 * n_shapelets;
    let sample_stride = n_channels * n_timepoints;

    // Allocate output; each sample writes its own row independently.
    let mut result = vec![0.0f64; n_samples * n_features];

    if n_samples == 1 {
        transform_sample(
            &x[..sample_stride],
            n_channels,
            n_timepoints,
            model,
            groups,
            &mut result,
        );
        return result;
    }

    // Process samples in parallel. Each chunk is one sample's feature row.
    result
        .par_chunks_mut(n_features)
        .enumerate()
        .for_each(|(i_sample, out_row)| {
            let sample = &x[i_sample * sample_stride..(i_sample + 1) * sample_stride];
            transform_sample(sample, n_channels, n_timepoints, model, &groups, out_row);
        });

    result
}

pub(crate) fn transform_scores(
    x: &[f64],
    n_samples: usize,
    n_channels: usize,
    n_timepoints: usize,
    model: &RdstModel,
    groups: &[ShapeletGroup],
    classifier: &PreparedRidgeParams,
) -> Vec<f64> {
    let n_rows = classifier.n_rows;
    let sample_stride = n_channels * n_timepoints;
    let mut result = vec![0.0f64; n_samples * n_rows];

    if n_samples == 1 {
        result.copy_from_slice(&classifier.intercept);
        transform_sample_scores(
            &x[..sample_stride],
            n_channels,
            n_timepoints,
            model,
            groups,
            classifier,
            &mut result,
        );
        return result;
    }

    result
        .par_chunks_mut(n_rows)
        .enumerate()
        .for_each(|(i_sample, scores)| {
            scores.copy_from_slice(&classifier.intercept);
            let sample = &x[i_sample * sample_stride..(i_sample + 1) * sample_stride];
            transform_sample_scores(
                sample,
                n_channels,
                n_timepoints,
                model,
                groups,
                classifier,
                scores,
            );
        });

    result
}

/// Process a single sample into `out_row` (length = `3 * n_shapelets`).
fn transform_sample(
    sample: &[f64],
    n_channels: usize,
    n_timepoints: usize,
    model: &RdstModel,
    groups: &[ShapeletGroup],
    out_row: &mut [f64],
) {
    let mut subs = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut sum = Vec::new();
    let mut sum2 = Vec::new();

    for group in groups {
        let length = group.length;
        let dilation = group.dilation;

        // Minimum number of timepoints needed for at least one subsequence:
        // the dilated span is (length − 1) * dilation + 1 timepoints.
        let min_needed = (length - 1) * dilation + 1;
        if n_timepoints < min_needed {
            // Window too short for this (length, dilation) pair.
            // Features for these shapelets stay at the initialised zero.
            continue;
        }
        let n_subs = n_timepoints - (length - 1) * dilation;

        // Compute raw subsequences (needed for both norm and non-norm shapelets).
        get_all_subsequences_into(
            &mut subs,
            sample,
            n_channels,
            n_timepoints,
            length,
            dilation,
        );

        // --- Non-normalised shapelets ---
        for &i_shp in &group.non_norm {
            let shp = &model.shapelets[i_shp];
            let (min_dist, arg_min, occurrence) = compute_shapelet_features(
                &subs,
                &shp.values,
                shp.threshold,
                n_subs,
                n_channels,
                length,
            );
            let off = 3 * i_shp;
            out_row[off] = min_dist;
            out_row[off + 1] = arg_min;
            out_row[off + 2] = occurrence;
        }

        // --- Normalised shapelets — compute mean/std once per (length, dilation) ---
        if !group.norm.is_empty() {
            sliding_mean_std_into(
                &mut means,
                &mut stds,
                &mut sum,
                &mut sum2,
                sample,
                n_channels,
                n_timepoints,
                length,
                dilation,
            );
            normalise_subsequences_in_place(&mut subs, &means, &stds, n_subs, n_channels, length);

            for &i_shp in &group.norm {
                let shp = &model.shapelets[i_shp];
                let (min_dist, arg_min, occurrence) = compute_shapelet_features(
                    &subs,
                    &shp.values,
                    shp.threshold,
                    n_subs,
                    n_channels,
                    length,
                );
                let off = 3 * i_shp;
                out_row[off] = min_dist;
                out_row[off + 1] = arg_min;
                out_row[off + 2] = occurrence;
            }
        }
    }
}

fn transform_sample_scores(
    sample: &[f64],
    n_channels: usize,
    n_timepoints: usize,
    model: &RdstModel,
    groups: &[ShapeletGroup],
    classifier: &PreparedRidgeParams,
    scores: &mut [f64],
) {
    let mut subs = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut sum = Vec::new();
    let mut sum2 = Vec::new();

    for group in groups {
        let length = group.length;
        let dilation = group.dilation;

        let min_needed = (length - 1) * dilation + 1;
        if n_timepoints < min_needed {
            continue;
        }
        let n_subs = n_timepoints - (length - 1) * dilation;
        get_all_subsequences_into(
            &mut subs,
            sample,
            n_channels,
            n_timepoints,
            length,
            dilation,
        );

        for &i_shp in &group.non_norm {
            let shp = &model.shapelets[i_shp];
            let (min_dist, arg_min, occurrence) = compute_shapelet_features(
                &subs,
                &shp.values,
                shp.threshold,
                n_subs,
                n_channels,
                length,
            );
            add_shapelet_scores(scores, classifier, i_shp, min_dist, arg_min, occurrence);
        }

        if !group.norm.is_empty() {
            sliding_mean_std_into(
                &mut means,
                &mut stds,
                &mut sum,
                &mut sum2,
                sample,
                n_channels,
                n_timepoints,
                length,
                dilation,
            );
            normalise_subsequences_in_place(&mut subs, &means, &stds, n_subs, n_channels, length);

            for &i_shp in &group.norm {
                let shp = &model.shapelets[i_shp];
                let (min_dist, arg_min, occurrence) = compute_shapelet_features(
                    &subs,
                    &shp.values,
                    shp.threshold,
                    n_subs,
                    n_channels,
                    length,
                );
                add_shapelet_scores(scores, classifier, i_shp, min_dist, arg_min, occurrence);
            }
        }
    }
}

#[inline]
fn add_shapelet_scores(
    scores: &mut [f64],
    classifier: &PreparedRidgeParams,
    i_shp: usize,
    min_dist: f64,
    arg_min: f64,
    occurrence: f64,
) {
    let feature_off = 3 * i_shp;
    for (row, score) in scores.iter_mut().enumerate().take(classifier.n_rows) {
        let coef_off = row * classifier.n_cols + feature_off;
        *score += min_dist * classifier.coef[coef_off]
            + arg_min * classifier.coef[coef_off + 1]
            + occurrence * classifier.coef[coef_off + 2];
    }
}

// ---------------------------------------------------------------------------
// Group bookkeeping
// ---------------------------------------------------------------------------

/// Precomputed index lists for a `(length, dilation)` group.
pub(crate) struct ShapeletGroup {
    /// Number of time points in the shapelets in this group.
    length: usize,
    /// Dilation factor used by the shapelets in this group.
    dilation: usize,
    /// Shapelet indices with `normalise = false`.
    non_norm: Vec<usize>,
    /// Shapelet indices with `normalise = true`.
    norm: Vec<usize>,
}

/// Builds the `(length, dilation) → ShapeletGroup` map from a model.
pub(crate) fn build_groups(model: &RdstModel) -> Vec<ShapeletGroup> {
    let mut map: HashMap<(usize, usize), ShapeletGroup> = HashMap::new();
    for (i, shp) in model.shapelets.iter().enumerate() {
        let entry = map
            .entry((shp.length, shp.dilation))
            .or_insert(ShapeletGroup {
                length: shp.length,
                dilation: shp.dilation,
                non_norm: Vec::new(),
                norm: Vec::new(),
            });
        if shp.normalise {
            entry.norm.push(i);
        } else {
            entry.non_norm.push(i);
        }
    }
    let mut groups: Vec<_> = map.into_values().collect();
    groups.sort_by_key(|group| (group.length, group.dilation));
    groups
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{RdstModel, RidgeParams, ScalerParams, ShapeletParams};

    fn make_model(shapelets: Vec<ShapeletParams>) -> RdstModel {
        let n = shapelets.len();
        RdstModel {
            version: "1.0".into(),
            n_shapelets: n,
            n_channels: 1,
            shapelets,
            scaler: ScalerParams {
                mean: vec![0.0; 3 * n],
                scale: vec![1.0; 3 * n],
            },
            classifier: RidgeParams {
                coef: vec![0.0; 3 * n],
                n_rows: 1,
                n_cols: 3 * n,
                intercept: vec![0.0],
                classes: vec!["a".into(), "b".into()],
            },
        }
    }

    #[test]
    fn output_shape() {
        // 1 shapelet, 1 channel, length=2, dilation=1
        let shp = ShapeletParams {
            values: vec![0.0, 1.0],
            n_channels: 1,
            length: 2,
            dilation: 1,
            threshold: 10.0,
            normalise: false,
            means: vec![0.0],
            stds: vec![0.0],
        };
        let model = make_model(vec![shp]);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 1 sample, 1 ch, 5 tp
        let out = transform(&x, 1, 1, 5, &model);
        assert_eq!(out.len(), 3); // 1 sample × 3 features
    }

    #[test]
    fn all_zeros_input_gives_finite() {
        let shp = ShapeletParams {
            values: vec![0.0, 1.0, 2.0],
            n_channels: 1,
            length: 3,
            dilation: 1,
            threshold: 10.0,
            normalise: true,
            means: vec![0.0],
            stds: vec![0.0],
        };
        let model = make_model(vec![shp]);
        let x = vec![0.0; 10]; // 1 sample, 1 ch, 10 tp
        let out = transform(&x, 1, 1, 10, &model);
        for v in &out {
            assert!(v.is_finite(), "expected finite, got {v}");
        }
    }

    #[test]
    fn two_different_samples_differ() {
        let shp = ShapeletParams {
            values: vec![1.0, 0.0],
            n_channels: 1,
            length: 2,
            dilation: 1,
            threshold: 5.0,
            normalise: false,
            means: vec![0.0],
            stds: vec![0.0],
        };
        let model = make_model(vec![shp]);
        // sample 0 = [0,0,0], sample 1 = [5,5,5]
        let x = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0];
        let out = transform(&x, 2, 1, 3, &model);
        // At least one feature should differ between the two samples.
        let same = (0..3).all(|i| (out[i] - out[3 + i]).abs() < 1e-12);
        assert!(
            !same,
            "two different samples should produce different features"
        );
    }
}
