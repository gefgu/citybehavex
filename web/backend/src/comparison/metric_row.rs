//! Mirrors `payload/legacy.py::_metric_row`: the common
//! `{filter_key, filter_label, metric_name, name, value, unit?}` shape used
//! by every metric list (`wasserstein`, `jsd`, `cpc`, `time_use`, `stvd`) in
//! `ChartPayload.metrics`.

use serde::Serialize;

use super::filters::FilterMeta;

#[derive(Debug, Clone, Serialize)]
pub struct MetricRow {
    pub filter_key: String,
    pub filter_label: String,
    pub metric_name: String,
    pub name: String,
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// Mirrors `_metric_row`: `None`/non-finite values are dropped (returns
/// `None`, matching the Python function's guard clause) rather than emitted
/// as `null`/`NaN` JSON.
pub fn metric_row(
    meta: &FilterMeta,
    metric_name: &str,
    value: Option<f64>,
    unit: &str,
) -> Option<MetricRow> {
    let value = value?;
    if !value.is_finite() {
        return None;
    }
    Some(MetricRow {
        filter_key: meta.key.clone(),
        filter_label: meta.label.clone(),
        metric_name: metric_name.to_string(),
        name: metric_name.to_string(),
        value,
        unit: if unit.is_empty() {
            None
        } else {
            Some(unit.to_string())
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::filters::FilterKind;
    use super::*;

    fn meta() -> FilterMeta {
        FilterMeta {
            key: "all".into(),
            label: "All".into(),
            kind: FilterKind::Base,
            start: None,
            end: None,
            hour_start: None,
            hour_end: None,
        }
    }

    #[test]
    fn drops_none_and_nonfinite_values() {
        assert!(metric_row(&meta(), "x", None, "").is_none());
        assert!(metric_row(&meta(), "x", Some(f64::NAN), "").is_none());
        assert!(metric_row(&meta(), "x", Some(f64::INFINITY), "").is_none());
    }

    #[test]
    fn keeps_finite_values_with_unit() {
        let row = metric_row(&meta(), "jump_lengths_km", Some(1.5), "km").unwrap();
        assert_eq!(row.value, 1.5);
        assert_eq!(row.unit.as_deref(), Some("km"));
        assert_eq!(row.metric_name, "jump_lengths_km");
        assert_eq!(row.name, "jump_lengths_km");
    }

    #[test]
    fn empty_unit_is_omitted() {
        let row = metric_row(&meta(), "cpc", Some(0.5), "").unwrap();
        assert!(row.unit.is_none());
    }
}
