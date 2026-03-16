//! Request/Response types for VelesDB REST API.
//!
//! Canonical DTOs are defined in `velesdb_core::api_types` and re-exported here.
//! Server-specific types that are not shared with other crates remain in this file.

// Re-export all canonical DTOs from core's api_types module.
// The `openapi` feature is enabled in this crate, providing ToSchema derives.
pub use velesdb_core::api_types::*;
