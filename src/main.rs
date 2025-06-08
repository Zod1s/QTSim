#![allow(unused)]

mod plots;
mod solver;
mod utils;
mod wiener;

use std::f64::consts::PI;

// use crate::plots;
use crate::solver::{Rk4, StochasticSolver, System};
use crate::utils::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use num_cpus;
use plotpy;
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

fn wmfme() -> utils::SolverResult<()> {
    let hamiltonian = na::Matrix2::new(1.0, 0.0, 0.0, -1.0).cast::<na::Complex<f64>>();
    let l = na::Matrix2::new(0.0, 1.0, 1.0, 0.0).cast::<na::Complex<f64>>();
    let f = na::Matrix2::new(
        na::Complex::<f64>::ZERO,
        -na::Complex::<f64>::I,
        na::Complex::<f64>::I,
        na::Complex::<f64>::ZERO,
    );
    let lhat = l - f * na::Complex::I;
    let hhat = hamiltonian + (f * l + l.adjoint() * f).scale(0.5);

    let system = QubitWisemanFME { h: hhat, l: lhat };

    // let mut xobs = plotpy::Curve::new();
    // xobs.set_label("X observable")
    //     .set_line_color("#FF0000")
    //     .set_line_width(3.0)
    //     .set_line_style("-");
    //
    // xobs.draw(t_out, &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>());
    //
    // let mut yobs = plotpy::Curve::new();
    // yobs.set_label("Y observable")
    //     .set_line_color("#00FF00")
    //     .set_line_width(3.0)
    //     .set_line_style("--");
    //
    // yobs.draw(t_out, &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>());
    //
    // let mut zobs = plotpy::Curve::new();
    // zobs.set_label("Z observable")
    //     .set_line_color("#0000FF")
    //     .set_line_width(3.0)
    //     .set_line_style("-");
    //
    // zobs.draw(t_out, &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>());
    //
    // let mut plot = plotpy::Plot::new();
    // plot.add(&xobs)
    //     .add(&yobs)
    //     .add(&zobs)
    //     .set_range(0.0, 2.0, -1.0, 1.0)
    //     .grid_labels_legend("time", "observable value");
    //
    // plot.show("tempimages")?;

    let mut sphere = plotpy::Surface::new();
    sphere
        .set_surf_color("#00000020")
        .draw_sphere(&[0.0, 0.0, 0.0], 1.0, 40, 40)?;

    let mut plot = plotpy::Plot::new();
    plot.add(&sphere).set_equal_axes(true);

    let colors = vec![
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    for i in 0..10 {
        let x0 = random_qubit_state();

        let mut solver = Rk4::new(system, 0.0, x0, 2.0, 0.001);
        solver.integrate()?;

        let (t_out, rho_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| to_bloch(rho))
            .collect::<Vec<BlochVector>>();
        let mut trajectory = plotpy::Curve::new();
        trajectory.set_line_color(colors[i]).draw_3d(
            &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        );

        plot.add(&trajectory);

        let mut start = plotpy::Curve::new();
        start
            .set_line_color(colors[i])
            .set_marker_style("o")
            .set_marker_size(10.0)
            .points_3d_begin()
            .points_3d_add(obsv[0][0], obsv[0][1], obsv[0][2])
            .points_3d_end();

        plot.add(&start);
    }

    // .add(&equator)
    // .add(&meridian)
    // .add(&meridian2);
    //     .set_equal_axes(true);
    plot.show("tempimages")?;

    Ok(())
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
