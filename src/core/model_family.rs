//! Model family definitions — capabilities, param counts, defaults, and UI metadata.
//!
//! Re-exports everything from `models.rs`, which parses `models.toml`
//! (the single source of truth for model metadata).
//!
//! The Python `arch_config.py` remains the source of truth for pipeline classes
//! and training flags — this module covers the user-facing / validation side.

pub use crate::core::models::*;
