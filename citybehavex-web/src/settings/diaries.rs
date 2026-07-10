//! Mirrors `citybehavex/llm_diaries/config.py::{SpecialDayConfig,DiariesConfig}`.

use chrono::{Datelike, NaiveDate, Weekday};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SpecialDayConfig {
    pub name: String,
    pub start_date: String,
    pub end_date: String,
    pub city_profile: String,
}

impl Default for SpecialDayConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            start_date: String::new(),
            end_date: String::new(),
            city_profile: String::new(),
        }
    }
}

impl SpecialDayConfig {
    fn start(&self) -> anyhow::Result<NaiveDate> {
        Ok(NaiveDate::parse_from_str(&self.start_date, "%Y-%m-%d")?)
    }

    fn end(&self) -> anyhow::Result<NaiveDate> {
        Ok(NaiveDate::parse_from_str(&self.end_date, "%Y-%m-%d")?)
    }

    pub fn contains(&self, day: NaiveDate) -> bool {
        match (self.start(), self.end()) {
            (Ok(start), Ok(end)) => start <= day && day <= end,
            _ => false,
        }
    }

    pub fn overlaps(&self, start: NaiveDate, end: NaiveDate) -> bool {
        match (self.start(), self.end()) {
            (Ok(self_start), Ok(self_end)) => self_start <= end && self_end >= start,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DiariesConfig {
    pub city_profile: String,
    pub city_profile_weekday: String,
    pub city_profile_weekend: String,
    pub representative_day: String,
    pub special_days: Vec<SpecialDayConfig>,
    pub allowed_purposes: Vec<String>,
    pub location_count_mu: f64,
    pub location_count_sigma: f64,
    pub max_locations: i64,
    pub max_one_location_diaries: Option<i64>,
    pub motif_exploration_rate: f64,
}

impl Default for DiariesConfig {
    fn default() -> Self {
        Self {
            city_profile: String::new(),
            city_profile_weekday: String::new(),
            city_profile_weekend: String::new(),
            representative_day: "2026-01-01".to_string(),
            special_days: Vec::new(),
            allowed_purposes: vec!["HOME".to_string(), "WORK".to_string(), "OTHER".to_string()],
            location_count_mu: 1.0,
            location_count_sigma: 0.5,
            max_locations: 6,
            max_one_location_diaries: None,
            motif_exploration_rate: 1.0,
        }
    }
}

impl DiariesConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !(self.location_count_sigma > 0.0) {
            anyhow::bail!("location_count_sigma must be > 0");
        }
        if !(1..=10).contains(&self.max_locations) {
            anyhow::bail!("max_locations must be in [1, 10]");
        }
        if let Some(v) = self.max_one_location_diaries {
            if v < 0 {
                anyhow::bail!("max_one_location_diaries must be >= 0");
            }
        }
        if !(0.0..=1.0).contains(&self.motif_exploration_rate) {
            anyhow::bail!("motif_exploration_rate must be in [0, 1]");
        }
        Ok(())
    }

    /// City profile for a day type, falling back to the shared one.
    pub fn profile_for(&self, day_type: &str) -> String {
        for special_day in &self.special_days {
            if special_day.name == day_type {
                return if special_day.city_profile.is_empty() {
                    self.city_profile.clone()
                } else {
                    special_day.city_profile.clone()
                };
            }
        }
        let specific = if day_type == "weekday" {
            &self.city_profile_weekday
        } else {
            &self.city_profile_weekend
        };
        if specific.is_empty() {
            self.city_profile.clone()
        } else {
            specific.clone()
        }
    }

    /// Day types needed to cover a date range: weekday/weekend plus any
    /// special days whose range overlaps `[start, end]`.
    pub fn day_types_for_range(&self, start: NaiveDate, end: NaiveDate) -> Vec<String> {
        let mut day_types = vec!["weekday".to_string(), "weekend".to_string()];
        day_types.extend(
            self.special_days
                .iter()
                .filter(|d| d.overlaps(start, end))
                .map(|d| d.name.clone()),
        );
        day_types
    }

    /// Day type for a single calendar date: a matching special day's name if
    /// `day` falls in its range, else the weekday/weekend calendar rule.
    pub fn resolve_day_type(&self, day: NaiveDate) -> String {
        for special_day in &self.special_days {
            if special_day.contains(day) {
                return special_day.name.clone();
            }
        }
        if day.weekday() == Weekday::Sat || day.weekday() == Weekday::Sun {
            "weekend".to_string()
        } else {
            "weekday".to_string()
        }
    }
}
