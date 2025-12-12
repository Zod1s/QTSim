#![allow(unused)]

mod examples;
mod kron;
mod lyapunov;
mod plots;
mod solver;
mod systems;
mod utils;
mod wiener;

use crate::utils::*;

fn main() -> utils::SolverResult<()> {
    // examples::parallel::parallel()
    let id = na::Matrix2::<na::Complex<f64>>::identity();
    let s1 = PAULIS
        .iter()
        .map(|pauli| pauli.kronecker(&id).kronecker(&id).kronecker(&id))
        .collect::<Vec<Operator<na::Const<16>>>>();

    let s2 = PAULIS
        .iter()
        .map(|pauli| id.kronecker(&pauli).kronecker(&id).kronecker(&id))
        .collect::<Vec<Operator<na::Const<16>>>>();

    let s3 = PAULIS
        .iter()
        .map(|pauli| id.kronecker(&id).kronecker(&pauli).kronecker(&id))
        .collect::<Vec<Operator<na::Const<16>>>>();

    let s4 = PAULIS
        .iter()
        .map(|pauli| id.kronecker(&id).kronecker(&id).kronecker(&pauli))
        .collect::<Vec<Operator<na::Const<16>>>>();

    let h = s1
        .iter()
        .zip(s2.iter())
        .map(|(p1, p2)| p1 * p2)
        .sum::<Operator<na::Const<16>>>()
        + s2.iter()
            .zip(s3.iter())
            .map(|(p1, p2)| p1 * p2)
            .sum::<Operator<na::Const<16>>>()
        + s3.iter()
            .zip(s4.iter())
            .map(|(p1, p2)| p1 * p2)
            .sum::<Operator<na::Const<16>>>()
        + s4.iter()
            .zip(s1.iter())
            .map(|(p1, p2)| p1 * p2)
            .sum::<Operator<na::Const<16>>>();

    println!("H: {}", h.map(|a| a.re));
    let eigen = h.symmetric_eigen();
    let zeroeigen =
        eigen.eigenvalues - na::SVector::<f64, 16>::from_element(1.) * eigen.eigenvalues.min();
    println!("Eigenvalues: {:.4}", eigen.eigenvalues);
    println!("Eigenvalues: {:.4}", zeroeigen);
    println!("Sum of eigenvalues: {:.4}", zeroeigen.sum());
    println!("Eigenvectors: {:.4}", eigen.eigenvectors.map(|v| v.re));

    Ok(())
}
