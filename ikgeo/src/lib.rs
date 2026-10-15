//! IK-Geo 6R kinematics. Vendored and adapted, see NOTICE and ../PLAN.md.
pub use nalgebra;

pub mod solutionset;

pub mod inverse;
pub mod inverse_kinematics;
pub mod reach;
pub mod robot;
pub mod subproblems;
pub mod track;

#[cfg(test)]
mod completeness;

#[cfg(test)]
mod correctness;

#[cfg(test)]
mod family_smoke;

#[cfg(test)]
mod profile_nj4;

#[cfg(test)]
mod timing;
