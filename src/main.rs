#![allow(unused)]

mod plots;
mod solver;
mod utils;
mod wiener;

use std::f64::consts::PI;

use crate::solver::{Rk4, StochasticSolver, System};
use crate::utils::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use num_cpus;
use rayon::prelude::*;
use std::process::Command;

fn main() -> utils::SolverResult<()> {
    wmfme()?;
    Ok(())
}

type Qubit = na::Matrix2<na::Complex<f64>>;

#[derive(Clone, Copy, Debug)]
struct QubitWisemanFME {
    h: Qubit,
    l: Qubit,
}

impl System<Qubit> for QubitWisemanFME {
    fn system(&self, t: f64, rho: &Qubit, drho: &mut Qubit) {
        *drho = -commutator(&self.h, rho) * na::Complex::I + self.l * rho * self.l.adjoint()
            - anticommutator(&(self.l.adjoint() * self.l), rho).scale(0.5);
    }
}

fn wmfme() -> utils::SolverResult<Vec<na::SVector<f64, 3>>> {
    let hamiltonian = na::Matrix2::new(
        na::Complex::<f64>::ONE,
        na::Complex::<f64>::ZERO,
        na::Complex::<f64>::ZERO,
        -na::Complex::<f64>::ONE,
    );
    let l = na::Matrix2::new(
        na::Complex::<f64>::ZERO,
        na::Complex::<f64>::ONE,
        na::Complex::<f64>::ONE,
        na::Complex::<f64>::ZERO,
    );
    let f = na::Matrix2::new(
        na::Complex::<f64>::ZERO,
        -na::Complex::<f64>::I,
        na::Complex::<f64>::I,
        na::Complex::<f64>::ZERO,
    );
    let lhat = l - f * na::Complex::I;
    let hhat = hamiltonian + (f * l + l.adjoint() * f).scale(0.5);

    let system = QubitWisemanFME { h: hhat, l: lhat };
    let x0 = na::Matrix2::new(
        na::Complex::<f64>::ONE,
        na::Complex::<f64>::ZERO,
        na::Complex::<f64>::ZERO,
        na::Complex::<f64>::ONE,
    )
    .scale(0.5);

    let mut solver = Rk4::new(system, 0.0, x0, 2.0, 0.001);
    solver.integrate()?;

    let (t_out, rho_out) = solver.results().get();

    rho_out
        .iter()
        .map(|rho| to_bloch(rho))
        .collect::<SolverResult<Vec<na::SVector<f64, 3>>>>()

    // plot.add_trace(
    //     plots::Scatter::new(ts.clone(), obs.iter().map(|x| x[0]).collect::<Vec<f64>>())
    //         .mode(plots::Mode::Lines)
    //         .line(
    //             plots::Line::default()
    //                 .width(3.)
    //                 .color(plots::NamedColor::Red),
    //         )
    //         .name("X observable"),
    // );
    //
    // plot.add_trace(
    //     plots::Scatter::new(ts.clone(), obs.iter().map(|x| x[1]).collect::<Vec<f64>>())
    //         .mode(plots::Mode::Lines)
    //         .line(
    //             plots::Line::default()
    //                 .width(3.)
    //                 .color(plots::NamedColor::Blue),
    //         )
    //         .name("Y observable"),
    // );
    //
    // plot.add_trace(
    //     plots::Scatter::new(ts.clone(), obs.iter().map(|x| x[2]).collect::<Vec<f64>>())
    //         .mode(plots::Mode::Lines)
    //         .line(
    //             plots::Line::default()
    //                 .width(3.)
    //                 .color(plots::NamedColor::Green),
    //         )
    //         .name("Z observable"),
    // );
    //
    // let layout = plots::Layout::default()
    //     .x_axis(
    //         plots::Axis::default()
    //             .show_grid(true)
    //             .title("Time")
    //             .range(vec![0., 2.0].to_vec()),
    //     )
    //     .y_axis(
    //         plots::Axis::default()
    //             .show_grid(true)
    //             .title("Observable value")
    //             .range(vec![-1., 1.].to_vec()),
    //     )
    //     .margin(plots::Margin::default().left(150))
    //     .title("Observable values")
    //     .font(plots::Font::default().size(30));
    //
    // plot.set_layout(layout);
    // plots::show_png(&plot, options);
}

// fn example() -> utils::SolverResult<()> {
//     rayon::ThreadPoolBuilder::new()
//         .num_threads(6)
//         .build_global()
//         .unwrap();
//
//     let omega = 1.; // angular frequency with which the Bloch vector rotates around the X-axis
//     let kappa = 0.01 * omega; // coupling between the Z-components of the Bloch vectors
//     let kappa1 = 0.005 * omega;
//     let kappa2 = kappa1;
//     let eta1 = 0.85;
//     let eta2 = 0.85;
//     let etas = vec![eta1, eta2];
//
//     let n_avg = 500;
//
//     let h = (PAULI_X.kronecker(&na::DMatrix::identity(2, 2))
//         + na::DMatrix::identity(2, 2).kronecker(&PAULI_X))
//     .scale(omega / 2.)
//         + PAULI_Z.kronecker(&PAULI_Z).scale(kappa);
//
//     let n = 5000; // number of integration steps per cycle
//     let dt = 2. * PI / (n as f64 * omega);
//     let n_cycles = 50;
//     let final_time = (2. * PI / omega) * n_cycles as f64;
//
//     let ls = vec![
//         PAULI_Z
//             .kronecker(&na::DMatrix::identity(2, 2))
//             .scale((2. * kappa1).sqrt()),
//         na::DMatrix::identity(2, 2)
//             .kronecker(&PAULI_Z)
//             .scale((2. * kappa2).sqrt()),
//     ];
//
//     let init_state1 = na::dmatrix![
//          na::Complex::ONE, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
//         na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
//         na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
//         na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO
//     ];
//
//     let init_state2 = na::DMatrix::identity(4, 4).scale(0.25);
//
//     let mut solver1 = StochasticSolver::new(&init_state1, &h, &ls, &etas, dt)?;
//     let mut solver2 = StochasticSolver::new(&init_state2, &h, &ls, &etas, dt)?;
//     let mut solvers = vec![solver1, solver2];
//
//     let colors = vec![plots::NamedColor::Red, plots::NamedColor::Green];
//     let title = "";
//     let xtitle = "Cycle";
//     let ytitle = "Purity";
//     let labels = vec!["Pure state", "Mixed state"];
//     let filepath = "plotters-doc-data/purity.png";
//
//     let options = plots::PlotOptions {
//         format: plots::Format::PNG,
//         width: 1200,
//         height: 800,
//         scale: 1.,
//     };
//
//     let mut plot = plots::Plot::new();
//     let num_samples = (final_time / solvers[0].dt).floor() as usize;
//
//     for (i, sol) in solvers.iter_mut().enumerate() {
//         let bar = ProgressBar::new(n_avg as u64).with_style(
//             ProgressStyle::default_bar()
//                 .template("Sample: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
//                 .unwrap(),
//         );
//
//         let purities = (0..n_avg)
//             .into_par_iter()
//             .progress_with(bar)
//             .fold_with(Ok(vec![0.; num_samples]), |acc, _| {
//                 let acc = acc?;
//                 let trajectory = sol.trajectory(final_time)?;
//                 Ok(acc
//                     .iter()
//                     .zip(
//                         &trajectory
//                             .0
//                             .iter()
//                             .map(|rho| (rho * rho).trace().re)
//                             .collect::<Vec<f64>>(),
//                     )
//                     .map(|(a, b)| a + b)
//                     .collect::<Vec<f64>>())
//             })
//             .collect::<Result<Vec<Vec<f64>>, SolverError>>()?
//             .into_par_iter()
//             .reduce(
//                 || vec![0.; num_samples],
//                 |acc, e| acc.iter().zip(e).map(|(a, b)| a + b).collect(),
//             )
//             .into_par_iter()
//             .map(|x| x / n_avg as f64)
//             .collect();
//
//         // let purities = (0..n_avg)
//         //     .into_par_iter()
//         //     .progress_with(bar1)
//         //     .map(|_| {
//         //         Ok(sol
//         //             .trajectory(final_time)?
//         //             .0
//         //             .iter()
//         //             .map(|rho| (rho * rho).trace().re)
//         //             .collect::<Vec<f64>>())
//         //     })
//         //     .collect::<Result<Vec<Vec<f64>>, SolverError>>()?
//         //     .into_par_iter()
//         //     .progress_with(bar2)
//         //     .reduce(
//         //         || vec![0.; num_samples],
//         //         |acc, e| acc.iter().zip(e).map(|(a, b)| a + b).collect(),
//         //     );
//
//         plot.add_trace(
//             plots::Scatter::new(
//                 (0..num_samples)
//                     .map(|x| x as f64 * n_cycles as f64 / num_samples as f64)
//                     .collect(),
//                 purities,
//             )
//             .mode(plots::Mode::Lines)
//             .line(plots::Line::default().width(3.).color(colors[i]))
//             .name(labels[i]),
//         );
//     }
//
//     let layout = plots::Layout::default()
//         .x_axis(
//             plots::Axis::default()
//                 .show_grid(true)
//                 .title(xtitle)
//                 .range(vec![0, n_cycles].to_vec()),
//         )
//         .y_axis(
//             plots::Axis::default()
//                 .show_grid(true)
//                 .title(ytitle)
//                 .range(vec![0., 1.].to_vec()),
//         )
//         .margin(plots::Margin::default().left(150))
//         .title(title)
//         .font(plots::Font::default().size(30));
//     // .plot_background_color(plots::NamedColor::White);
//
//     plot.set_layout(layout);
//
//     plots::show_png(&plot, options);
//
//     Ok(())
// }
