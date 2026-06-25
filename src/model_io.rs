//! JSON deserialization for RDST model files.
//!
//! Supports both camelCase (`nShapelets`) and snake_case (`n_shapelets`) field
//! names, matching the Python aeon export helper output.

use serde_json::Value;

use crate::{
    errors::ClassifierError,
    model::{RdstModel, RidgeParams, ScalerParams, ShapeletParams},
};

/// Parses an [`RdstModel`] from a JSON string.
pub fn from_json(json_str: &str) -> Result<RdstModel, ClassifierError> {
    let v: Value = serde_json::from_str(json_str)?;
    parse_model(&v)
}

// ---------------------------------------------------------------------------
// Internal parsers
// ---------------------------------------------------------------------------

fn parse_model(v: &Value) -> Result<RdstModel, ClassifierError> {
    let version = v["version"]
        .as_str()
        .ok_or_else(|| invalid("missing 'version' field"))?
        .to_owned();

    let n_shapelets = get_int(v, &["nShapelets", "n_shapelets"])? as usize;
    let n_channels = get_int(v, &["nChannels", "n_channels"])? as usize;

    let shapelets_v = v
        .get("shapelets")
        .ok_or_else(|| invalid("missing 'shapelets' field"))?;
    let shapelets = parse_shapelets(shapelets_v, n_shapelets, n_channels)?;

    let scaler_v = v
        .get("scaler")
        .ok_or_else(|| invalid("missing 'scaler' field"))?;
    let scaler = parse_scaler(scaler_v)?;

    let classifier_v = v
        .get("classifier")
        .ok_or_else(|| invalid("missing 'classifier' field"))?;
    let classifier = parse_classifier(classifier_v)?;

    Ok(RdstModel {
        version,
        n_shapelets,
        n_channels,
        shapelets,
        scaler,
        classifier,
    })
}

fn parse_shapelets(
    v: &Value,
    n_shapelets: usize,
    n_channels: usize,
) -> Result<Vec<ShapeletParams>, ClassifierError> {
    let values_json = v["values"]
        .as_array()
        .ok_or_else(|| invalid("shapelets.values must be an array"))?;
    let lengths = int_array(&v["lengths"])?;
    let dilations = int_array(&v["dilations"])?;
    let thresholds = f64_array(&v["thresholds"])?;
    let normalise_arr = bool_array(&v["normalise"])?;
    let means_json = v["means"]
        .as_array()
        .ok_or_else(|| invalid("shapelets.means must be an array"))?;
    let stds_json = v["stds"]
        .as_array()
        .ok_or_else(|| invalid("shapelets.stds must be an array"))?;

    let mut shapelets = Vec::with_capacity(n_shapelets);

    for i in 0..n_shapelets {
        let length = lengths[i] as usize;

        // values_json[i] is List[n_channels][length]
        let channel_values = values_json[i]
            .as_array()
            .ok_or_else(|| invalid(&format!("shapelets.values[{i}] must be an array")))?;

        let mut flat_values = vec![0.0f64; n_channels * length];
        for c in 0..n_channels {
            let row = channel_values[c]
                .as_array()
                .ok_or_else(|| invalid(&format!("shapelets.values[{i}][{c}] must be an array")))?;
            for j in 0..length {
                flat_values[c * length + j] = as_f64(&row[j])?;
            }
        }

        let means = f64_array(&means_json[i])?;
        let stds = f64_array(&stds_json[i])?;

        shapelets.push(ShapeletParams {
            values: flat_values,
            n_channels,
            length,
            dilation: dilations[i] as usize,
            threshold: thresholds[i],
            normalise: normalise_arr[i],
            means,
            stds,
        });
    }

    Ok(shapelets)
}

fn parse_scaler(v: &Value) -> Result<ScalerParams, ClassifierError> {
    Ok(ScalerParams {
        mean: f64_array(&v["mean"])?,
        scale: f64_array(&v["scale"])?,
    })
}

fn parse_classifier(v: &Value) -> Result<RidgeParams, ClassifierError> {
    let coef_json = v["coef"]
        .as_array()
        .ok_or_else(|| invalid("classifier.coef must be an array"))?;

    let n_rows = coef_json.len();
    let n_cols = coef_json[0]
        .as_array()
        .ok_or_else(|| invalid("classifier.coef[0] must be an array"))?
        .len();

    let mut coef = vec![0.0f64; n_rows * n_cols];
    for r in 0..n_rows {
        let row = coef_json[r]
            .as_array()
            .ok_or_else(|| invalid(&format!("classifier.coef[{r}] must be an array")))?;
        for c in 0..n_cols {
            coef[r * n_cols + c] = as_f64(&row[c])?;
        }
    }

    let intercept = f64_array(&v["intercept"])?;

    let classes_json = v["classes"]
        .as_array()
        .ok_or_else(|| invalid("classifier.classes must be an array"))?;
    let classes: Vec<String> = classes_json
        .iter()
        .map(|c| {
            c.as_str()
                .map(|s| s.to_owned())
                .ok_or_else(|| invalid("class label must be a string"))
        })
        .collect::<Result<_, _>>()?;

    Ok(RidgeParams {
        coef,
        n_rows,
        n_cols,
        intercept,
        classes,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_int(v: &Value, keys: &[&str]) -> Result<i64, ClassifierError> {
    for &key in keys {
        if let Some(n) = v.get(key).and_then(|x| x.as_i64()) {
            return Ok(n);
        }
    }
    Err(invalid(&format!(
        "missing integer field (tried: {})",
        keys.join(", ")
    )))
}

fn f64_array(v: &Value) -> Result<Vec<f64>, ClassifierError> {
    let arr = v
        .as_array()
        .ok_or_else(|| invalid("expected a JSON array"))?;
    arr.iter().map(as_f64).collect()
}

fn int_array(v: &Value) -> Result<Vec<i64>, ClassifierError> {
    let arr = v
        .as_array()
        .ok_or_else(|| invalid("expected a JSON array"))?;
    arr.iter()
        .map(|x| x.as_i64().ok_or_else(|| invalid("expected integer")))
        .collect()
}

fn bool_array(v: &Value) -> Result<Vec<bool>, ClassifierError> {
    let arr = v
        .as_array()
        .ok_or_else(|| invalid("expected a JSON array"))?;
    arr.iter()
        .map(|x| x.as_bool().ok_or_else(|| invalid("expected boolean")))
        .collect()
}

fn as_f64(v: &Value) -> Result<f64, ClassifierError> {
    v.as_f64()
        .ok_or_else(|| invalid("expected a floating-point number"))
}

fn invalid(msg: &str) -> ClassifierError {
    ClassifierError::InvalidModel {
        msg: msg.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_BINARY: &str = r#"{
        "version": "1.0",
        "n_shapelets": 1,
        "n_channels": 2,
        "shapelets": {
            "values":     [[[1.0, 2.0], [3.0, 4.0]]],
            "lengths":    [2],
            "dilations":  [1],
            "thresholds": [5.0],
            "normalise":  [true],
            "means":      [[0.0, 0.0]],
            "stds":       [[1.0, 1.0]]
        },
        "scaler": {
            "mean":  [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0]
        },
        "classifier": {
            "coef":      [[0.1, 0.2, 0.3]],
            "intercept": [0.0],
            "classes":   ["a", "b"]
        }
    }"#;

    #[test]
    fn parse_minimal_binary() {
        let model = from_json(MINIMAL_BINARY).unwrap();
        assert_eq!(model.version, "1.0");
        assert_eq!(model.n_shapelets, 1);
        assert_eq!(model.n_channels, 2);
        assert_eq!(model.shapelets.len(), 1);
        let shp = &model.shapelets[0];
        assert_eq!(shp.length, 2);
        assert_eq!(shp.dilation, 1);
        assert_eq!(shp.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(model.classifier.classes, vec!["a", "b"]);
        assert!(model.classifier.is_binary());
    }

    #[test]
    fn camel_case_keys() {
        let json = r#"{
            "version": "1.0",
            "nShapelets": 1,
            "nChannels": 2,
            "shapelets": {
                "values":     [[[1.0, 2.0], [3.0, 4.0]]],
                "lengths":    [2],
                "dilations":  [1],
                "thresholds": [5.0],
                "normalise":  [false],
                "means":      [[0.0, 0.0]],
                "stds":       [[0.0, 0.0]]
            },
            "scaler":     { "mean": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0] },
            "classifier": { "coef": [[1.0, 2.0, 3.0]], "intercept": [0.0], "classes": ["x", "y"] }
        }"#;
        let model = from_json(json).unwrap();
        assert_eq!(model.n_shapelets, 1);
    }
}
