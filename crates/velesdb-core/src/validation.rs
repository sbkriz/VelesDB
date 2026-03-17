//! Unified dimension validation helpers.
//!
//! Centralizes dimension range checks and mismatch validation used across
//! collection creation, CRUD, and search paths.

use crate::error::{Error, Result};

/// Minimum valid vector dimension.
pub const MIN_DIMENSION: usize = 1;

/// Maximum valid vector dimension (65,536 — covers all known embedding models).
pub const MAX_DIMENSION: usize = 65_536;

/// Validates that a vector dimension is within the allowed range.
///
/// # Errors
///
/// Returns [`Error::InvalidDimension`] if `dimension` is outside
/// [`MIN_DIMENSION`]`..=`[`MAX_DIMENSION`].
pub fn validate_dimension(dimension: usize) -> Result<()> {
    if !(MIN_DIMENSION..=MAX_DIMENSION).contains(&dimension) {
        return Err(Error::InvalidDimension {
            dimension,
            min: MIN_DIMENSION,
            max: MAX_DIMENSION,
        });
    }
    Ok(())
}

/// Validates that a vector's actual dimension matches the expected dimension.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] if `actual != expected`.
pub fn validate_dimension_match(expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::DimensionMismatch { expected, actual });
    }
    Ok(())
}
