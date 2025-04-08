#![allow(unused)]
// rendere Wiener un riferimento dietro ARC in modo da poter fare tutto in parallelo, così si
// accelera molto quando devo fare la media

mod solver;
mod utils;
mod wiener;

use std::f64::consts::PI;

use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use plotters::prelude::*;

use crate::solver::Solver;
use crate::utils::*;
use std::process::Command;

fn main() -> Result<(), SolverError> {
    let omega = 100.; // angular frequency with which the Bloch vector rotates around the X-axis
    let kappa = 0.01 * omega; // coupling between the Z-components of the Bloch vectors
    let kappa1 = 0.005 * omega;
    let kappa2 = kappa1;
    let eta1 = 0.85;
    let eta2 = 0.85;
    let etas = vec![eta1, eta2];

    let n_avg = 500;

    let h = (PAULI_X.kronecker(&na::DMatrix::identity(2, 2))
        + na::DMatrix::identity(2, 2).kronecker(&PAULI_X))
    .scale(omega / 2.)
        + PAULI_Z.kronecker(&PAULI_Z).scale(kappa);

    let n = 5000; // number of integration steps per cycle
    let dt = 2. * PI / (n as f64 * omega);
    let n_cycles = 50;
    let final_time = (2. * PI / omega) * n_cycles as f64;

    let ls = vec![
        PAULI_Z
            .kronecker(&na::DMatrix::identity(2, 2))
            .scale((2. * kappa1).sqrt()),
        na::DMatrix::identity(2, 2)
            .kronecker(&PAULI_Z)
            .scale((2. * kappa2).sqrt()),
    ];

    let init_state1 = na::dmatrix![
         na::Complex::ONE, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
        na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
        na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO;
        na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO, na::Complex::ZERO
    ];

    let init_state2 = na::DMatrix::identity(4, 4).scale(0.25);

    let mut solver1 = Solver::new(&init_state1, &h, &ls, &etas, dt)?;
    let mut solver2 = Solver::new(&init_state2, &h, &ls, &etas, dt)?;
    let solvers = vec![solver1, solver2];

    purity_graph(
        solvers,
        vec![RED, GREEN],
        "",
        "Cycle",
        "Purity",
        vec!["Pure state", "Completely mixed state"],
        final_time,
        n_avg,
        n_cycles,
        "plotters-doc-data/purity.png",
    )
}

fn purity_graph(
    mut solvers: Vec<Solver>,
    colors: Vec<RGBColor>,
    title: &str,
    xtitle: &str,
    ytitle: &str,
    labels: Vec<&str>,
    final_time: f64,
    n_avg: usize,
    n_cycles: usize,
    filepath: &str,
) -> Result<(), utils::SolverError> {
    let root = BitMapBackend::new(filepath, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let num_samples = (final_time / solvers[0].dt).floor() as usize;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f64..n_cycles as f64, 0f64..1f64)?;

    chart.configure_mesh().draw()?;

    for (i, sol) in solvers.iter_mut().enumerate() {
        let mut purities = vec![0_f64; num_samples];

        let bar = ProgressBar::new(n_avg as u64).with_style(
            ProgressStyle::default_bar()
                .template("Sample: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
                .unwrap(),
        );

        for _ in 0..n_avg {
            let trajectory = sol.trajectory(final_time)?.0;
            bar.inc(1);
            purities = purities
                .iter()
                .zip(
                    &trajectory
                        .iter()
                        .map(|rho| (rho * rho).trace().re)
                        .collect::<Vec<f64>>(),
                )
                .map(|(a, b)| a + b)
                .collect();
        }

        bar.finish();

        chart
            .draw_series(LineSeries::new(
                (0..num_samples).map(|x| {
                    (
                        n_cycles as f64 * x as f64 / num_samples as f64,
                        purities[x] / n_avg as f64,
                    )
                }),
                colors[i].clone(),
            ))?
            .label(labels[i]);
        // .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i].clone()));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Command::new("wslview")
        .arg(filepath)
        .output()
        .expect("Could not open the image");

    Ok(())
}
