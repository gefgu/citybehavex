//! Mirrors `web/backend/app/filters.py`: day-type/time-of-day/special-day
//! filter metadata and application, shared across every payload section.

use super::util::to_datetime_expr;
use polars::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    Base,
    Day,
    Time,
    DateRange,
}

#[derive(Debug, Clone, Serialize)]
pub struct FilterMeta {
    pub key: String,
    pub label: String,
    #[serde(skip)]
    pub kind: FilterKind,
    #[serde(skip)]
    pub start: Option<String>,
    #[serde(skip)]
    pub end: Option<String>,
    #[serde(skip)]
    pub hour_start: Option<i32>,
    #[serde(skip)]
    pub hour_end: Option<i32>,
}

/// The `{"key": ..., "label": ...}` shape the frontend actually sees --
/// mirrors `legacy.py::_public_filter` (drops the internal `kind`/`start`/`end`).
#[derive(Debug, Clone, Serialize)]
pub struct PublicFilter {
    pub key: String,
    pub label: String,
}

impl FilterMeta {
    pub fn public(&self) -> PublicFilter {
        PublicFilter {
            key: self.key.clone(),
            label: self.label.clone(),
        }
    }
}

pub fn filters() -> Vec<FilterMeta> {
    vec![
        FilterMeta {
            key: "all".into(),
            label: "All".into(),
            kind: FilterKind::Base,
            start: None,
            end: None,
            hour_start: None,
            hour_end: None,
        },
        FilterMeta {
            key: "weekday".into(),
            label: "Weekday".into(),
            kind: FilterKind::Day,
            start: None,
            end: None,
            hour_start: None,
            hour_end: None,
        },
        FilterMeta {
            key: "weekend".into(),
            label: "Weekend".into(),
            kind: FilterKind::Day,
            start: None,
            end: None,
            hour_start: None,
            hour_end: None,
        },
    ]
}

pub fn time_filters() -> Vec<FilterMeta> {
    vec![
        FilterMeta {
            key: "morning".into(),
            label: "Morning".into(),
            kind: FilterKind::Time,
            start: None,
            end: None,
            hour_start: Some(6),
            hour_end: Some(12),
        },
        FilterMeta {
            key: "afternoon".into(),
            label: "Afternoon".into(),
            kind: FilterKind::Time,
            start: None,
            end: None,
            hour_start: Some(12),
            hour_end: Some(18),
        },
        FilterMeta {
            key: "evening".into(),
            label: "Evening".into(),
            kind: FilterKind::Time,
            start: None,
            end: None,
            hour_start: Some(18),
            hour_end: Some(24),
        },
        FilterMeta {
            key: "night".into(),
            label: "Night".into(),
            kind: FilterKind::Time,
            start: None,
            end: None,
            hour_start: Some(0),
            hour_end: Some(6),
        },
    ]
}

/// One `(name, start_date, end_date)` triple, mirrors `ComparisonContext.special_days`.
#[derive(Debug, Clone)]
pub struct SpecialDay {
    pub name: String,
    pub start_date: String,
    pub end_date: String,
}

fn title_case(s: &str) -> String {
    s.split(' ')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Mirrors `filters.py::_special_day_filters`.
pub fn special_day_filters(special_days: &[SpecialDay]) -> Vec<FilterMeta> {
    special_days
        .iter()
        .map(|sd| FilterMeta {
            key: sd.name.clone(),
            label: title_case(&sd.name.replace('_', " ")),
            kind: FilterKind::DateRange,
            start: Some(sd.start_date.clone()),
            end: Some(sd.end_date.clone()),
            hour_start: None,
            hour_end: None,
        })
        .collect()
}

/// Mirrors `filters.py::_filter_df`. `datetime_col` absent/not-found or
/// `meta.key == "all"` returns `df` unchanged.
pub fn filter_df(
    df: &DataFrame,
    datetime_col: Option<&str>,
    meta: &FilterMeta,
) -> anyhow::Result<DataFrame> {
    let Some(datetime_col) = datetime_col else {
        return Ok(df.clone());
    };
    if meta.key == "all" || df.column(datetime_col).is_err() {
        return Ok(df.clone());
    }

    let schema = df.schema();
    let dt_expr = to_datetime_expr(&schema, datetime_col);

    let mask_expr = match meta.kind {
        FilterKind::Base => return Ok(df.clone()),
        FilterKind::Day => {
            // Polars `.dt().weekday()` is ISO: 1=Monday..7=Sunday.
            let is_weekday = dt_expr.clone().dt().weekday().lt(lit(6));
            if meta.key == "weekend" {
                is_weekday.not()
            } else {
                is_weekday
            }
        }
        FilterKind::DateRange => {
            let start = meta.start.as_deref().unwrap_or_default();
            let end = meta.end.as_deref().unwrap_or_default();
            let day = dt_expr.clone().dt().truncate(lit("1d"));
            day.clone()
                .gt_eq(lit(start).str().to_datetime(
                    Some(TimeUnit::Microseconds),
                    None,
                    StrptimeOptions {
                        strict: false,
                        ..Default::default()
                    },
                    lit("raise"),
                ))
                .and(day.lt_eq(lit(end).str().to_datetime(
                    Some(TimeUnit::Microseconds),
                    None,
                    StrptimeOptions {
                        strict: false,
                        ..Default::default()
                    },
                    lit("raise"),
                )))
        }
        FilterKind::Time => {
            let hour = dt_expr.clone().dt().hour();
            hour.clone()
                .gt_eq(lit(meta.hour_start.unwrap_or(0)))
                .and(hour.lt(lit(meta.hour_end.unwrap_or(24))))
        }
    };

    Ok(df
        .clone()
        .lazy()
        .filter(mask_expr.fill_null(lit(false)))
        .collect()?)
}

/// Mirrors `filters.py::_filter_visits`: `filter_df` fixed to the
/// `start_timestamp` column (the shape `_visits_for_comparison` produces).
pub fn filter_visits(
    visits: Option<&DataFrame>,
    meta: &FilterMeta,
) -> anyhow::Result<Option<DataFrame>> {
    match visits {
        None => Ok(None),
        Some(v) => Ok(Some(filter_df(v, Some("start_timestamp"), meta)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        df![
            "dt" => [
                "2026-01-05T08:00:00", // Monday
                "2026-01-10T08:00:00", // Saturday
                "2026-01-11T20:00:00", // Sunday, night-ish hour
            ],
        ]
        .unwrap()
    }

    #[test]
    fn all_filter_is_identity() {
        let df = sample_df();
        let all = filters().into_iter().find(|f| f.key == "all").unwrap();
        let out = filter_df(&df, Some("dt"), &all).unwrap();
        assert_eq!(out.height(), 3);
    }

    #[test]
    fn weekday_and_weekend_partition_correctly() {
        let df = sample_df();
        let weekday = filters().into_iter().find(|f| f.key == "weekday").unwrap();
        let weekend = filters().into_iter().find(|f| f.key == "weekend").unwrap();
        assert_eq!(filter_df(&df, Some("dt"), &weekday).unwrap().height(), 1);
        assert_eq!(filter_df(&df, Some("dt"), &weekend).unwrap().height(), 2);
    }

    #[test]
    fn time_filter_selects_hour_range() {
        let df = sample_df();
        let evening = time_filters()
            .into_iter()
            .find(|f| f.key == "evening")
            .unwrap();
        assert_eq!(filter_df(&df, Some("dt"), &evening).unwrap().height(), 1);
    }

    #[test]
    fn special_day_filter_selects_date_range() {
        let df = sample_df();
        let sd = special_day_filters(&[SpecialDay {
            name: "emergency".into(),
            start_date: "2026-01-10".into(),
            end_date: "2026-01-11".into(),
        }]);
        let out = filter_df(&df, Some("dt"), &sd[0]).unwrap();
        assert_eq!(out.height(), 2);
        assert_eq!(sd[0].label, "Emergency");
    }

    #[test]
    fn missing_datetime_column_returns_unchanged() {
        let df = sample_df();
        let weekday = filters().into_iter().find(|f| f.key == "weekday").unwrap();
        let out = filter_df(&df, Some("nonexistent"), &weekday).unwrap();
        assert_eq!(out.height(), 3);
    }
}
