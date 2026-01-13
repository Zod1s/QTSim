#![allow(unused)]

mod dataplots;
mod examples;
mod kron;
mod lyapunov;
mod plots;
mod solver;
mod systems;
mod utils;
mod wiener;

use crate::utils::*;
use rayon::prelude::*;
const NUMTHREADS: usize = 14;

// Consider optimising \beta and \varepsilon for the original controller and compute the optimal
// window to reduce the variance under a certain threshold
fn main() -> utils::SolverResult<()> {
    // examples::actualfeed::actualfeed()
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(NUMTHREADS.min(num_cpus::get()).max(1))
    //     .build_global()
    //     .expect("Could not create threadpool");
    //
    // rayon::scope(|s| {
    //     s.spawn(|s| {
    //         examples::parallel::parallel_3d(false);
    //     });
    //     s.spawn(|s| {
    //         examples::parallel::parallel_heis(false);
    //     });
    // });

    dataplots::plot("./heis.csv")

    // let id = na::Matrix2::<na::Complex<f64>>::identity();
    // let s1 = PAULIS
    //     .iter()
    //     .map(|pauli| pauli.kronecker(&id).kronecker(&id).kronecker(&id))
    //     .collect::<Vec<Operator<na::U16>>>();
    //
    // let s2 = PAULIS
    //     .iter()
    //     .map(|pauli| id.kronecker(&pauli).kronecker(&id).kronecker(&id))
    //     .collect::<Vec<Operator<na::U16>>>();
    //
    // let s3 = PAULIS
    //     .iter()
    //     .map(|pauli| id.kronecker(&id).kronecker(&pauli).kronecker(&id))
    //     .collect::<Vec<Operator<na::U16>>>();
    //
    // let s4 = PAULIS
    //     .iter()
    //     .map(|pauli| id.kronecker(&id).kronecker(&id).kronecker(&pauli))
    //     .collect::<Vec<Operator<na::U16>>>();
    //
    // let weights = vec![1., 1., 1.0];
    // let ferromag = -1.;
    // let hm = 2.0;
    //
    // let h = -s1
    //     .iter()
    //     .zip(s2.iter())
    //     .zip(weights.iter())
    //     .map(|((p1, p2), w)| p1 * p2.scale(ferromag * w))
    //     .sum::<Operator<na::U16>>()
    //     - s2.iter()
    //         .zip(s3.iter())
    //         .zip(weights.iter())
    //         .map(|((p1, p2), w)| p1 * p2.scale(ferromag * w))
    //         .sum::<Operator<na::U16>>()
    //     - s3.iter()
    //         .zip(s4.iter())
    //         .zip(weights.iter())
    //         .map(|((p1, p2), w)| p1 * p2.scale(ferromag * w))
    //         .sum::<Operator<na::U16>>()
    //     - (s1[2] + s2[2] + s3[2] + s4[2]).scale(hm)
    //     - s4.iter()
    //         .zip(s1.iter())
    //         .zip(weights.iter())
    //         .map(|((p1, p2), w)| p1 * p2.scale(ferromag * w))
    //         .sum::<Operator<na::U16>>();
    //
    // println!("H: {}", h.map(|a| a.re));
    // let eigen = h.symmetric_eigen();
    // println!("Eigenvalues: {:.4}", eigen.eigenvalues);
    // println!("Sum of eigenvalues: {:.4}", eigen.eigenvalues.sum());
    // println!("Eigenvectors: {:.4}", eigen.eigenvectors.map(|v| v.re));
    //
    // Ok(())
}
