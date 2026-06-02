//! # difftop-rs
//!
//! A library for differential topology: smooth manifolds, tangent bundles,
//! differential forms, Stokes' theorem, transversality, degree theory,
//! and Euler characteristic via Poincaré-Hopf.

pub mod agent_dynamics;
pub mod degree;
pub mod differential_form;
pub mod euler;
pub mod smooth_manifold;
pub mod stokes;
pub mod tangent;
pub mod transversality;
pub mod vector_field;

pub use agent_dynamics::*;
pub use degree::*;
pub use differential_form::*;
pub use euler::*;
pub use smooth_manifold::*;
pub use stokes::*;
pub use tangent::*;
pub use transversality::*;
pub use vector_field::*;
