//! One quality decision shared by world generation and renderer profiles.
//!
//! The URL adapter parses `?spec=` into this application-owned value once.
//! Generation and presentation then derive their independent concrete
//! settings from the same decision, preventing case or fallback drift.

use std::{error::Error, fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QualityProfile {
    /// Standard workloads intended for integrated and mobile-class GPUs.
    #[default]
    Low,
    /// Finer generation and larger rendering workloads for high-end systems.
    High,
}

impl QualityProfile {
    pub const ALL: [Self; 2] = [Self::Low, Self::High];

    /// Parses the stable `low`/`high` vocabulary case-insensitively.
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        Self::ALL
            .into_iter()
            .find(|quality| value.eq_ignore_ascii_case(quality.query_value()))
    }

    pub const fn query_value(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::High => "high",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Low => "Standard",
            Self::High => "High",
        }
    }
}

impl fmt::Display for QualityProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.query_value())
    }
}

impl FromStr for QualityProfile {
    type Err = ParseQualityProfileError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or(ParseQualityProfileError)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseQualityProfileError;

impl fmt::Display for ParseQualityProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("expected quality profile `low` or `high`")
    }
}

impl Error for ParseQualityProfileError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_quality_vocabulary_is_case_insensitive() {
        for quality in QualityProfile::ALL {
            let value = quality.query_value();
            assert_eq!(QualityProfile::parse(value), Some(quality));
            assert_eq!(
                QualityProfile::parse(&value.to_ascii_uppercase()),
                Some(quality)
            );
            assert_eq!(value.parse(), Ok(quality));
        }
        assert_eq!(QualityProfile::parse("ultra"), None);
    }
}
