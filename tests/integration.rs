//! Integration tests that load the same JSON fixture files used by the Dart
//! test suite and verify bit-exact predictions / near-exact probabilities.
//!
//! Fixture path is resolved relative to `CARGO_MANIFEST_DIR` so the tests work
//! from any working directory. If the fixture files are not present (e.g. the
//! `dart_rdst_classifier` submodule was not checked out) the tests are skipped
//! with a warning rather than failing.

use std::io::Read;
use std::path::PathBuf;

use rdst_classifier::RdstClassifier;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the path to the Dart fixture directory.
fn fixture_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .unwrap()
        .join("dart_rdst_classifier")
        .join("test")
        .join("fixtures")
}

/// Reads a fixture file, returning `None` if it doesn't exist.
fn read_fixture(name: &str) -> Option<String> {
    let path = fixture_dir().join(name);
    if path.exists() {
        Some(std::fs::read_to_string(&path).expect("fixture read error"))
    } else {
        eprintln!("SKIP: fixture not found: {}", path.display());
        None
    }
}

/// Flattens a nested `serde_json::Value` of shape `[samples][channels][timepoints]`
/// into a `Vec<f64>` in sample-major, channel-major order.
fn to_flat(raw: &serde_json::Value) -> (Vec<f64>, usize, usize, usize) {
    let samples = raw.as_array().unwrap();
    let n_samples = samples.len();
    let n_channels = samples[0].as_array().unwrap().len();
    let n_timepoints = samples[0][0].as_array().unwrap().len();

    let mut out = Vec::with_capacity(n_samples * n_channels * n_timepoints);
    for sample in samples.iter().take(n_samples) {
        for channel in sample.as_array().unwrap().iter().take(n_channels) {
            for value in channel.as_array().unwrap().iter().take(n_timepoints) {
                out.push(value.as_f64().unwrap());
            }
        }
    }
    (out, n_samples, n_channels, n_timepoints)
}

// ---------------------------------------------------------------------------
// Binary end-to-end
// ---------------------------------------------------------------------------

#[test]
fn binary_classes_match() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("binary_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["binary"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let expected: Vec<String> = fix["classes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();

    assert_eq!(clf.classes(), expected.as_slice());
}

#[test]
fn binary_predict_matches_python() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("binary_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["binary"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);

    let preds = clf.predict(&x, n_s, n_c, n_t).unwrap();
    let expected: Vec<String> = fix["expected_predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();

    assert_eq!(preds, expected, "binary predictions must match Python");
}

#[test]
fn binary_proba_shape() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("binary_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["binary"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);

    let pr = clf.predict_proba(&x, n_s, n_c, n_t).unwrap();
    assert_eq!(pr.len(), n_s * 2, "binary proba shape: n_samples × 2");
}

#[test]
fn binary_proba_rows_sum_to_one() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("binary_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["binary"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);

    let pr = clf.predict_proba(&x, n_s, n_c, n_t).unwrap();
    for i in 0..n_s {
        let row_sum = pr[i * 2] + pr[i * 2 + 1];
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "binary sample {i} proba row must sum to 1, got {row_sum}"
        );
    }
}

#[test]
fn binary_proba_matches_python() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("binary_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["binary"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);

    let pr = clf.predict_proba(&x, n_s, n_c, n_t).unwrap();
    let exp = fix["expected_probas"].as_array().unwrap();

    for i in 0..n_s {
        let exp_row = exp[i].as_array().unwrap();
        for c in 0..2 {
            let expected_p = exp_row[c].as_f64().unwrap();
            assert!(
                (pr[i * 2 + c] - expected_p).abs() < 1e-4,
                "binary sample={i} class={c}: got {} expected {}",
                pr[i * 2 + c],
                expected_p
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Multiclass end-to-end
// ---------------------------------------------------------------------------

#[test]
fn multiclass_predict_matches_python() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("multiclass_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["multiclass"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);

    let preds = clf.predict(&x, n_s, n_c, n_t).unwrap();
    let expected: Vec<String> = fix["expected_predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();

    assert_eq!(preds, expected, "multiclass predictions must match Python");
}

#[test]
fn multiclass_proba_rows_sum_to_one() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("multiclass_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["multiclass"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);
    let n_classes = clf.classes().len();

    let pr = clf.predict_proba(&x, n_s, n_c, n_t).unwrap();
    for i in 0..n_s {
        let row_sum: f64 = (0..n_classes).map(|c| pr[i * n_classes + c]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "multiclass sample {i} proba row must sum to 1, got {row_sum}"
        );
    }
}

#[test]
fn multiclass_proba_matches_python() {
    let Some(preds_json) = read_fixture("expected_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("multiclass_model.json") else {
        return;
    };

    let fixtures: serde_json::Value = serde_json::from_str(&preds_json).unwrap();
    let fix = &fixtures["multiclass"];

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);
    let n_classes = clf.classes().len();

    let pr = clf.predict_proba(&x, n_s, n_c, n_t).unwrap();
    let exp = fix["expected_probas"].as_array().unwrap();

    for i in 0..n_s {
        let exp_row = exp[i].as_array().unwrap();
        for c in 0..n_classes {
            let expected_p = exp_row[c].as_f64().unwrap();
            assert!(
                (pr[i * n_classes + c] - expected_p).abs() < 1e-4,
                "multiclass sample={i} class={c}: got {} expected {}",
                pr[i * n_classes + c],
                expected_p
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Integration model — 574 real samples
// ---------------------------------------------------------------------------

#[test]
fn integration_predict_matches_python_all_574() {
    let Some(preds_json) = read_fixture("integration_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("integration_model.json") else {
        return;
    };

    let fix: serde_json::Value = serde_json::from_str(&preds_json).unwrap();

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);

    assert_eq!(n_s, 574, "integration fixture should have 574 samples");

    let preds = clf.predict(&x, n_s, n_c, n_t).unwrap();
    let expected: Vec<String> = fix["expected_predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();

    assert_eq!(
        preds, expected,
        "all 574 integration predictions must match Python"
    );
}

#[test]
fn integration_proba_rows_sum_to_one() {
    let Some(preds_json) = read_fixture("integration_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("integration_model.json") else {
        return;
    };

    let fix: serde_json::Value = serde_json::from_str(&preds_json).unwrap();

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);
    let n_classes = clf.classes().len();

    let pr = clf.predict_proba(&x, n_s, n_c, n_t).unwrap();
    for i in 0..n_s {
        let row_sum: f64 = (0..n_classes).map(|c| pr[i * n_classes + c]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "integration sample {i} proba row must sum to 1, got {row_sum}"
        );
    }
}

#[test]
fn integration_proba_matches_python() {
    let Some(preds_json) = read_fixture("integration_predictions.json") else {
        return;
    };
    let Some(model_json) = read_fixture("integration_model.json") else {
        return;
    };

    let fix: serde_json::Value = serde_json::from_str(&preds_json).unwrap();

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_s, n_c, n_t) = to_flat(&fix["test_X"]);
    let n_classes = clf.classes().len();

    let pr = clf.predict_proba(&x, n_s, n_c, n_t).unwrap();
    let exp = fix["expected_probas"].as_array().unwrap();

    for i in 0..n_s {
        let exp_row = exp[i].as_array().unwrap();
        for c in 0..n_classes {
            let expected_p = exp_row[c].as_f64().unwrap();
            assert!(
                (pr[i * n_classes + c] - expected_p).abs() < 1e-4,
                "integration sample={i} class={c}: got {} expected {}",
                pr[i * n_classes + c],
                expected_p
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Dimension mismatch error
// ---------------------------------------------------------------------------

#[test]
fn dimension_mismatch_returns_error() {
    let Some(model_json) = read_fixture("binary_model.json") else {
        return;
    };
    let clf = RdstClassifier::from_json(&model_json).unwrap();
    // Wrong slice length
    let err = clf.predict(&[1.0, 2.0], 1, 4, 10);
    assert!(err.is_err(), "wrong slice length should produce an error");
}

#[test]
fn wrong_channel_count_returns_error() {
    let Some(model_json) = read_fixture("binary_model.json") else {
        return;
    };
    let clf = RdstClassifier::from_json(&model_json).unwrap();
    // model has 4 channels; we pass 3
    let x = vec![0.0; 180];
    let err = clf.predict(&x, 1, 3, 60);
    assert!(err.is_err(), "wrong channel count should produce an error");
}

// ---------------------------------------------------------------------------
// .tar.gz loading + RSF1 sliding-window classification
// ---------------------------------------------------------------------------

/// Assets directory (`test/assets/` inside dart_rdst_classifier).
fn assets_dir() -> PathBuf {
    fixture_dir()
        .parent()
        .unwrap() // test/
        .join("assets")
}

/// Parses an RSF1 file.
/// Layout of output: channel-major — `data[c * n_timepoints + t]`.
/// Extract the value of the "content" JSON string field without a full JSON
/// parse, so files with invalid Unicode in other fields (author, comment) still
/// work.  The content field only contains digits, commas, minus signs, and
/// `\n`/`\r` escapes, so a simple scan suffices.
fn extract_rsf_content(text: &str) -> Option<String> {
    let marker = "\"content\":\"";
    let start = text.find(marker)? + marker.len();
    let rest = &text[start..];
    let mut result = String::with_capacity(rest.len());
    let mut chars = rest.chars();
    loop {
        match chars.next()? {
            '"' => break,
            '\\' => match chars.next()? {
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                _ => {} // skip other escape sequences
            },
            c => result.push(c),
        }
    }
    Some(result)
}

fn parse_rsf1(bytes: &[u8], n_channels: usize) -> Option<(Vec<f64>, usize)> {
    if bytes.len() < 4 || &bytes[0..4] != b"RSF1" {
        return None;
    }
    let mut gz = flate2::read::GzDecoder::new(&bytes[4..]);
    let mut json_bytes = Vec::new();
    gz.read_to_end(&mut json_bytes).ok()?;
    // Use lossy UTF-8 so files with invalid bytes in other fields don't fail.
    let json_str = String::from_utf8_lossy(&json_bytes);
    let content = extract_rsf_content(&json_str)?;

    let rows: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    let n_timepoints = rows.len();
    if n_timepoints == 0 {
        return None;
    }

    let mut flat = vec![0.0f64; n_channels * n_timepoints];
    for (t, row) in rows.iter().enumerate() {
        let fields: Vec<&str> = row.split(',').collect();
        if fields.len() < n_channels {
            return None;
        }
        for c in 0..n_channels {
            flat[c * n_timepoints + t] = fields[c].trim().parse().ok()?;
        }
    }
    Some((flat, n_timepoints))
}

/// Extracts a window slice (channel-major) from a channel-major flat array.
fn slice_window(
    src: &[f64],
    n_channels: usize,
    n_timepoints_total: usize,
    start: usize,
    length: usize,
) -> Vec<f64> {
    let mut w = vec![0.0f64; n_channels * length];
    for c in 0..n_channels {
        w[c * length..c * length + length].copy_from_slice(
            &src[c * n_timepoints_total + start..c * n_timepoints_total + start + length],
        );
    }
    w
}

#[test]
fn tar_gz_model_loads_successfully() {
    let path = assets_dir().join("models").join("model.tar.gz");
    if !path.exists() {
        eprintln!("SKIP: model.tar.gz not found");
        return;
    }
    let bytes = std::fs::read(&path).unwrap();
    let clf = RdstClassifier::from_tar_gz(&bytes).expect("model.tar.gz should load");
    assert_eq!(clf.model().n_channels, 25);
    assert_eq!(clf.model().n_shapelets, 10_000);
    assert!(!clf.classes().is_empty());
}

#[test]
fn rsf_files_classify_to_valid_classes() {
    let model_path = assets_dir().join("models").join("model.tar.gz");
    let rsf_dir = assets_dir().join("test_files");
    if !model_path.exists() || !rsf_dir.exists() {
        eprintln!("SKIP: production assets not found");
        return;
    }

    let bytes = std::fs::read(&model_path).unwrap();
    let clf = RdstClassifier::from_tar_gz(&bytes).unwrap();
    let n_channels = clf.model().n_channels;
    let valid_classes: std::collections::HashSet<&str> =
        clf.classes().iter().map(|s| s.as_str()).collect();

    const WINDOW_SIZE: usize = 5;
    const WINDOW_STRIDE: usize = 20;

    let mut rsf_files: Vec<_> = std::fs::read_dir(&rsf_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "rsf").unwrap_or(false))
        .collect();
    rsf_files.sort_by_key(|e| e.path());
    assert!(!rsf_files.is_empty(), "no RSF files found");

    let mut total_windows = 0usize;
    for entry in &rsf_files {
        let file_bytes = std::fs::read(entry.path()).unwrap();
        let (flat, n_timepoints) = parse_rsf1(&file_bytes, n_channels).expect("RSF1 parse failed");

        if n_timepoints <= WINDOW_SIZE {
            let pred = clf.predict(&flat, 1, n_channels, n_timepoints).unwrap();
            assert!(
                valid_classes.contains(pred[0].as_str()),
                "{}: unexpected class '{}'",
                entry.path().display(),
                pred[0]
            );
            total_windows += 1;
        } else {
            let mut start = 0;
            loop {
                if start >= n_timepoints {
                    break;
                }
                let end = (start + WINDOW_SIZE).min(n_timepoints);
                let length = end - start;
                let w = slice_window(&flat, n_channels, n_timepoints, start, length);
                let pred = clf.predict(&w, 1, n_channels, length).unwrap();
                assert!(
                    valid_classes.contains(pred[0].as_str()),
                    "{}: unexpected class '{}'",
                    entry.path().display(),
                    pred[0]
                );
                total_windows += 1;
                if end == n_timepoints {
                    break;
                }
                start += WINDOW_STRIDE;
            }
        }
    }

    println!(
        "RSF test: classified {} windows across {} files",
        total_windows,
        rsf_files.len()
    );
    assert!(total_windows > 0);
}
