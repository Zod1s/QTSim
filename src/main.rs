#![allow(unused)]

mod solver;
mod utils;
mod wiener;

use std::f64::consts::PI;

use crate::solver::Solver;
use crate::utils::*;

fn main() {
    let omega = 1.; // angular frequency with which the Bloch vector rotates around the X-axis
    let kappa = 0.01 * omega; // coupling between the Z-components of the Bloch vectors
    let kappa1 = 0.005 * omega;
    let kappa2 = kappa1;
    let eta1 = 0.85;
    let eta2 = 0.85;
    let etas = vec![eta1, eta2];

    let h = (PAULI_X.kronecker(&na::Matrix2::identity())
        + na::Matrix2::identity().kronecker(&PAULI_X))
    .scale(omega / 2.)
        + PAULI_Z.kronecker(&PAULI_Z).scale(kappa);

    let n = 5000.; // number of integration steps per cycle
    let dt = 2. * PI / (n * omega);

    let ls = vec![
        PAULI_Z
            .kronecker(&na::Matrix2::identity())
            .scale((2. * kappa1).sqrt()),
        na::Matrix2::identity()
            .kronecker(&PAULI_Z)
            .scale((2. * kappa2).sqrt()),
    ];

    let init_state = na::matrix![
        na::Complex::ONE, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
        na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
        na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
        na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO
    ];

    let solver = Solver::new(init_state, h, ls, etas, dt);
}
