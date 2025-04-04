#![allow(unused)]

mod solver;
mod utils;
mod wiener;

use std::f64::consts::PI;

use dataviz::figure::datasets::dataset::Dataset;
use dataviz::figure::{
    canvas::pixelcanvas::PixelCanvas, configuration::figureconfig::FigureConfig,
    datasets::cartesiangraphdataset::CartesianDataset, display::winop::Winop,
    drawers::drawer::Drawer, figuretypes::quadrant1graph::Quadrant1Graph,
    utilities::linetype::LineType,
};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};

use crate::solver::Solver;
use crate::utils::*;

fn main() -> Result<(), Error> {
    let omega = 0.01; // angular frequency with which the Bloch vector rotates around the X-axis
    let kappa = 0.01 * omega; // coupling between the Z-components of the Bloch vectors
    let kappa1 = 0.005 * omega;
    let kappa2 = kappa1;
    let eta1 = 0.85;
    let eta2 = 0.85;
    let etas = vec![eta1, eta2];

    let n_avg = 100;

    let h = (PAULI_X.kronecker(&na::DMatrix::identity(2, 2))
        + na::DMatrix::identity(2, 2).kronecker(&PAULI_X))
    .scale(omega / 2.)
        + PAULI_Z.kronecker(&PAULI_Z).scale(kappa);

    let n = 5000; // number of integration steps per cycle
    let dt = 2. * PI / (n as f64 * omega);
    let n_cycles = 50;
    let final_time = 2. * PI / omega * n_cycles as f64;

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

    let mut figure_config = FigureConfig::default();
    figure_config.set_font_paths(
        "/home/lore/.local/share/fonts/CaskaydiaCoveNerdFontPropo-Regular.ttf".to_owned(),
        "/home/lore/.local/share/fonts/CaskaydiaCoveNerdFontPropo-Regular.ttf".to_owned(),
    );
    let mut pixel_canvas = PixelCanvas::new(800, 600, [255, 255, 255], 80);

    let mut graph = Quadrant1Graph::new("Purity", "Cycle", "Purity", figure_config.clone());

    let mut purity_line1 = CartesianDataset::new([0, 0, 255], "purity1", LineType::Solid);
    let mut purity_line2 = CartesianDataset::new([255, 0, 0], "purity2", LineType::Solid);

    let mut solver1 = Solver::new(&init_state1, &h, &ls, &etas, dt)?;
    let mut purities1 = vec![0_f64; (final_time / dt).floor() as usize];
    let bar1 = ProgressBar::new(n_avg as u64).with_style(
        ProgressStyle::default_bar()
            .template("Sample: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );
    for _ in 0..n_avg {
        let trajectory = solver1.trajectory(final_time)?;
        purities1 = purities1
            .iter()
            .zip(
                &trajectory
                    .iter()
                    .map(|rho| (rho * rho).trace().re)
                    .collect::<Vec<f64>>(),
            )
            .map(|(a, b)| a + b)
            .collect();
        bar1.inc(1);
    }

    bar1.finish();

    let mut solver2 = Solver::new(&init_state2, &h, &ls, &etas, dt)?;
    let mut purities2 = vec![0_f64; (final_time / dt).floor() as usize];
    let bar2 = ProgressBar::new(n_avg as u64).with_style(
        ProgressStyle::default_bar()
            .template("Sample: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );
    for _ in 0..n_avg {
        let trajectory = solver2.trajectory(final_time)?;
        purities2 = purities2
            .iter()
            .zip(
                &trajectory
                    .iter()
                    .map(|rho| (rho * rho).trace().re)
                    .collect::<Vec<f64>>(),
            )
            .map(|(a, b)| a + b)
            .collect();
        bar2.inc(1);
    }

    bar2.finish();

    let num_points = purities1.len();
    for x in 0..num_points {
        let y1 = purities1[x] / n_avg as f64;
        let y2 = purities2[x] / n_avg as f64;
        let t = x as f64 * dt / (2. * PI);
        purity_line1.add_point((t, y1));
        purity_line2.add_point((t, y2));
    }

    graph.add_dataset(purity_line1);
    graph.add_dataset(purity_line2);

    graph.draw(&mut pixel_canvas);

    Winop::display_interactive(&mut pixel_canvas, &mut graph, "Purity");

    Ok(())
}
