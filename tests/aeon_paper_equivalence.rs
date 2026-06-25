use rdst_classifier::RdstClassifier;

fn to_flat(raw: &serde_json::Value) -> (Vec<f64>, usize, usize, usize) {
    let samples = raw.as_array().unwrap();
    let n_samples = samples.len();
    let n_channels = samples[0].as_array().unwrap().len();
    let n_timepoints = samples[0][0].as_array().unwrap().len();

    let mut out = Vec::with_capacity(n_samples * n_channels * n_timepoints);
    for sample in samples {
        for channel in sample.as_array().unwrap() {
            for value in channel.as_array().unwrap() {
                out.push(value.as_f64().unwrap());
            }
        }
    }

    (out, n_samples, n_channels, n_timepoints)
}

#[test]
fn predictions_match_aeon_on_paper_rsf_fixture() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/aeon_paper_equivalence.json")).unwrap();
    let model_json = serde_json::to_string(&fixture["model"]).unwrap();
    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_samples, n_channels, n_timepoints) = to_flat(&fixture["test_X"]);

    let expected: Vec<String> = fixture["expected_predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().to_owned())
        .collect();

    let preds = clf
        .predict(&x, n_samples, n_channels, n_timepoints)
        .unwrap();
    assert_eq!(preds, expected);
}

#[test]
fn aeon_paper_fixture_probabilities_are_well_formed() {
    // Aeon's default RDSTClassifier predict_proba is one-hot because the fitted
    // RidgeClassifierCV has no probability API. Rust keeps its existing
    // score-based probability output, so this test validates shape and bounds.
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/aeon_paper_equivalence.json")).unwrap();
    let model_json = serde_json::to_string(&fixture["model"]).unwrap();
    let clf = RdstClassifier::from_json(&model_json).unwrap();
    let (x, n_samples, n_channels, n_timepoints) = to_flat(&fixture["test_X"]);

    let n_classes = clf.classes().len();
    let probas = clf
        .predict_proba(&x, n_samples, n_channels, n_timepoints)
        .unwrap();

    assert_eq!(probas.len(), n_samples * n_classes);
    for sample in 0..n_samples {
        let row_sum: f64 = (0..n_classes)
            .map(|class| probas[sample * n_classes + class])
            .sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-6,
            "sample {sample} probability row sums to {row_sum}"
        );
        for class in 0..n_classes {
            let proba = probas[sample * n_classes + class];
            assert!(proba.is_finite());
            assert!((0.0..=1.0).contains(&proba));
        }
    }
}
