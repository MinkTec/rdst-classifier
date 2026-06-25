//! Criterion benchmarks for the RDST classifier.
//!
//! Mirrors the Dart benchmark scenarios and adds a real-world RSF
//! sliding-window scenario using the production `model.tar.gz`.
//!
//! Run with: `cargo bench -p rdst-classifier`
//!
//! # Scenarios
//! 1. `batch/predict/1000`        — 1000-sample tiled batch (integration model)
//! 2. `batch/predict_proba/1000`  — same, with probabilities
//! 3. `single/predict`            — single-sample repeated (integration model)
//! 4. `single/predict_proba`      — single-sample repeated (integration model)
//! 5. `production/predict_single_window` — single window from each RSF file using
//!    the production `model.tar.gz` (10 000 shapelets, 25 channels)

use std::io::Read;
use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rdst_classifier::RdstClassifier;

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("dart_rdst_classifier")
        .join("test")
        .join("fixtures")
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("dart_rdst_classifier")
        .join("test")
        .join("assets")
}

// ---------------------------------------------------------------------------
// Integration-model bench data (574 samples → tiled to 1000)
// ---------------------------------------------------------------------------

struct BatchBenchData {
    clf: RdstClassifier,
    x_1000: Vec<f64>,
    x_1: Vec<f64>,
    n_channels: usize,
    n_timepoints: usize,
}

fn load_batch_bench_data() -> Option<BatchBenchData> {
    let dir = fixture_dir();
    let model_path = dir.join("integration_model.json");
    let preds_path = dir.join("integration_predictions.json");
    if !model_path.exists() || !preds_path.exists() {
        eprintln!("SKIP integration RDST bench: fixtures not found");
        return None;
    }

    let model_json = std::fs::read_to_string(model_path).expect("integration_model.json");
    let preds_json = std::fs::read_to_string(preds_path).expect("integration_predictions.json");

    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let fix: serde_json::Value = serde_json::from_str(&preds_json).unwrap();

    let samples = fix["test_X"].as_array().unwrap();
    let n_src = samples.len();
    let n_channels = samples[0].as_array().unwrap().len();
    let n_timepoints = samples[0][0].as_array().unwrap().len();
    let stride = n_channels * n_timepoints;

    let mut src = vec![0.0f64; n_src * stride];
    for s in 0..n_src {
        for c in 0..n_channels {
            for t in 0..n_timepoints {
                src[s * stride + c * n_timepoints + t] = samples[s][c][t].as_f64().unwrap();
            }
        }
    }

    let target = 1000;
    let mut x_1000 = vec![0.0f64; target * stride];
    for i in 0..target {
        let src_start = (i % n_src) * stride;
        x_1000[i * stride..i * stride + stride]
            .copy_from_slice(&src[src_start..src_start + stride]);
    }

    let x_1 = src[0..stride].to_vec();

    Some(BatchBenchData {
        clf,
        x_1000,
        x_1,
        n_channels,
        n_timepoints,
    })
}

// ---------------------------------------------------------------------------
// RSF1 parser
//
// Format:
//   bytes 0..4  = b"RSF1"  (magic)
//   bytes 4..   = gzip-compressed JSON
//   JSON["content"] = newline-separated CSV rows (timepoints × channels)
//
// Returns (flat_data, n_channels, n_timepoints) in channel-major order
// (same layout expected by RdstClassifier).
// ---------------------------------------------------------------------------

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
                _ => {}
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

// ---------------------------------------------------------------------------
// Production bench data (model.tar.gz + RSF files)
// ---------------------------------------------------------------------------

/// One window worth of data ready to classify.
struct Window {
    x: Vec<f64>,
    n_channels: usize,
    n_timepoints: usize,
}

struct ProductionBenchData {
    clf: RdstClassifier,
    windows: Vec<Window>,
}

fn load_production_bench_data() -> Option<ProductionBenchData> {
    let model_path = assets_dir().join("models").join("model.tar.gz");
    let rsf_dir = assets_dir().join("test_files");
    if !model_path.exists() || !rsf_dir.exists() {
        eprintln!("SKIP production bench: assets not found");
        return None;
    }

    let model_bytes = std::fs::read(&model_path).ok()?;
    let clf = RdstClassifier::from_tar_gz(&model_bytes).ok()?;
    let n_channels = clf.model().n_channels;

    let window_size = 5usize;
    let window_stride = 20usize;
    let mut windows = Vec::new();

    let mut entries: Vec<_> = std::fs::read_dir(&rsf_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "rsf").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in &entries {
        let bytes = std::fs::read(entry.path()).ok()?;
        let (flat, n_timepoints) = parse_rsf1(&bytes, n_channels)?;
        // Build all windows from this file
        if n_timepoints <= window_size {
            windows.push(Window {
                x: flat.clone(),
                n_channels,
                n_timepoints,
            });
        } else {
            let mut start = 0;
            loop {
                if start >= n_timepoints {
                    break;
                }
                let end = (start + window_size).min(n_timepoints);
                let length = end - start;
                let mut w = vec![0.0f64; n_channels * length];
                for c in 0..n_channels {
                    w[c * length..c * length + length]
                        .copy_from_slice(&flat[c * n_timepoints + start..c * n_timepoints + end]);
                }
                windows.push(Window {
                    x: w,
                    n_channels,
                    n_timepoints: length,
                });
                if end == n_timepoints {
                    break;
                }
                start += window_stride;
            }
        }
    }

    Some(ProductionBenchData { clf, windows })
}

// ---------------------------------------------------------------------------
// Bench 1 & 2: batch predict / predict_proba
// ---------------------------------------------------------------------------

fn bench_batch_predict(c: &mut Criterion) {
    let Some(data) = load_batch_bench_data() else {
        return;
    };
    let n = 1000usize;

    let mut group = c.benchmark_group("batch");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function(BenchmarkId::new("predict", n), |b| {
        b.iter(|| {
            data.clf
                .predict(
                    black_box(&data.x_1000),
                    black_box(n),
                    black_box(data.n_channels),
                    black_box(data.n_timepoints),
                )
                .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("predict_proba", n), |b| {
        b.iter(|| {
            data.clf
                .predict_proba(
                    black_box(&data.x_1000),
                    black_box(n),
                    black_box(data.n_channels),
                    black_box(data.n_timepoints),
                )
                .unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 3 & 4: single-sample predict / predict_proba
// ---------------------------------------------------------------------------

fn bench_single_predict(c: &mut Criterion) {
    let Some(data) = load_batch_bench_data() else {
        return;
    };

    let mut group = c.benchmark_group("single");
    group.throughput(Throughput::Elements(1));

    group.bench_function("predict", |b| {
        b.iter(|| {
            data.clf
                .predict(
                    black_box(&data.x_1),
                    black_box(1),
                    black_box(data.n_channels),
                    black_box(data.n_timepoints),
                )
                .unwrap()
        })
    });

    group.bench_function("predict_proba", |b| {
        b.iter(|| {
            data.clf
                .predict_proba(
                    black_box(&data.x_1),
                    black_box(1),
                    black_box(data.n_channels),
                    black_box(data.n_timepoints),
                )
                .unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Bench 5: production model — sliding-window over all RSF files
// ---------------------------------------------------------------------------

fn bench_production_rsf(c: &mut Criterion) {
    let Some(data) = load_production_bench_data() else {
        eprintln!("SKIP bench_production_rsf: fixtures not available");
        return;
    };

    let n_windows = data.windows.len();
    let mut group = c.benchmark_group("production");
    group.throughput(Throughput::Elements(n_windows as u64));

    // Measure single-window latency (cycle through all windows)
    group.bench_function("predict_single_window", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let w = &data.windows[idx % n_windows];
            idx += 1;
            data.clf
                .predict(
                    black_box(&w.x),
                    black_box(1),
                    black_box(w.n_channels),
                    black_box(w.n_timepoints),
                )
                .unwrap()
        })
    });

    group.bench_function("predict_proba_single_window", |b| {
        let mut idx = 0usize;
        b.iter(|| {
            let w = &data.windows[idx % n_windows];
            idx += 1;
            data.clf
                .predict_proba(
                    black_box(&w.x),
                    black_box(1),
                    black_box(w.n_channels),
                    black_box(w.n_timepoints),
                )
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_predict,
    bench_single_predict,
    bench_production_rsf
);
criterion_main!(benches);
