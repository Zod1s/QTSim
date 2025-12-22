use crate::plots::{self, constrainedlayout};
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use std::sync::{Arc, Mutex};

const NUMTHREADS: usize = 8;

pub fn output() -> SolverResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(NUMTHREADS.min(num_cpus::get()).max(1))
        .build_global()
        .expect("Could not access plot");
    // let h = PAULI_Z;
    // let l = PAULI_Z;
    // let hc = QubitOperator::zeros();
    // let f0 = PAULI_X;
    // let f1 = QubitOperator::zeros();

    // let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    // let x0bloch = to_bloch(&x0)?;
    let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    // .scale(2.);
    // let f0 = na::Matrix3::zeros();
    let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let f1 = na::Matrix3::zeros();

    let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast();
    let x0 = na::Vector3::new(1., 1., 1.).cast();
    let x0 = x0 * x0.conjugate().transpose().scale(1. / 3.);

    let num_tries = 10;
    let final_time: f64 = 40.0;
    let dt = 0.0001;
    let decimation = 40;

    let k = 5000;
    let avg_fac = (k as f64) * dt;

    // let beta = 0.9;
    // let epsilon = 4.;
    let delta = 3.;
    let gamma = 0.2 * delta;
    let y1 = -2.;
    let epsilon = delta * 1.;
    let beta = 0.9;
    let normal = Normal::standard();
    let tf = (normal.inverse_cdf((beta + 1.) / 2.) / epsilon).powi(2);

    // let a = 0.01; let b = 0.02;

    // let rho1 = na::Matrix2::new(1.0, 0.0, 0.0, 0.0).cast();
    // let rho2 = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    // let y1 = ((l + l.adjoint()) * rho1).trace().re;
    // let y2 = ((l + l.adjoint()) * rho2).trace().re;
    // let gamma = 0.2;
    // let ub1 = (y1 - y2).abs() * gamma;
    // let ub2 = (y1 - y2).abs() * gamma / 2.;
    let ub = 2. * delta - gamma + y1;
    let lb = 2. * delta - 2. * gamma + y1;

    // let mut plots = Vec::with_capacity(num_tries);
    // for i in 0..num_tries {
    //     let mut plot = plotpy::Plot::new();
    //     plots.push(plot);
    // }
    let plot = Arc::new(Mutex::new(plotpy::Plot::new()));

    let colors = [
        "#358763", "#E78A18", "#00FBFF", "#3E00FF", "#E64500", "#FFEE00", "#0078FF", "#FF0037",
        "#E1FF00", "#00FF00",
    ];

    let bar = ProgressBar::new(num_tries as u64).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .expect("Could not access plot"),
    );

    (0..num_tries).into_par_iter().progress_with(bar).map(|i| {
        let mut rng = StdRng::seed_from_u64(i as u64);
        let mut system = systems::multilevelcompletefeedback::Feedback2::new(
            h, l, hc, f0, f1, y1, k, delta, gamma, beta, epsilon, &mut rng,
        );
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate();

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| fidelity(rho, &rhod))
            .collect::<Vec<f64>>();

        let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
            .map(|i| t_out[i * decimation])
            .collect();

        let obsv_dec = (0..obsv.len() / decimation)
            .map(|i| obsv[i * decimation])
            .collect();

        let mut fidline = plotpy::Curve::new();
        fidline
            .set_line_color(colors[i])
            .draw(&t_out_dec, &obsv_dec);

        plot.lock().expect("Could not access plot").add(&fidline);
    });

    // for i in 0..num_tries {
    // constrainedlayout(&format!("Images/plot{i}"), &mut plots[i], true)?;
    // plots[i].show("tempimages")?;
    // }
    plot.lock()
        .expect("Could not access plot")
        .show("tempimages")?;

    Ok(())
}

// for i in 0..num_tries {
//     bar.inc(1);
//     let mut rng = StdRng::seed_from_u64(i as u64);
//     // let mut system = systems::idealqubitcompletefeedback::QubitFeedback::new(
//     //     h, l, hc, f0, f1, y1, y2, gamma, &mut rng,
//     // );
//     // let mut system = systems::qubitcompletefeedback::QubitFeedback2::new(
//     //     h, l, hc, f0, f1, y1, y2, k, gamma, beta, epsilon, &mut rng,
//     // );
//     let mut system = systems::multilevelcompletefeedback::Feedback2::new(
//         h, l, hc, f0, f1, y1, k, delta, gamma, beta, epsilon, &mut rng,
//     );
//     let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
//     solver.integrate()?;
//
//     let (t_out, rho_out, dy_out) = solver.results().get();
//
//     // let mut y = vec![0.; dy_out.len()];
//     // for i in 1..dy_out.len() {
//     //     y[i] = y[i - 1] + dy_out[i - 1];
//     // }
//     //
//     // let bar_y: Vec<f64> = rho_out
//     //     .iter()
//     //     .map(|rho| 2. * (l * rho).trace().re)
//     //     .collect();
//     //
//     // let mut hat_y = vec![0.; dy_out.len()];
//     //
//     // for i in 1..hat_y.len() {
//     //     hat_y[i] = hat_y[i - 1] + dy_out[i - 1];
//     // }
//     //
//     // for i in 0..hat_y.len() {
//     //     if t_out[i] < tf {
//     //         hat_y[i] = 0.
//     //     } else {
//     //         hat_y[i] = hat_y[i] / t_out[i];
//     //     }
//     // }
//     //
//     // let mut wind_y = vec![0.; dy_out.len()];
//     // let mut wind_y2 = vec![0.; dy_out.len()];
//     //
//     // for i in 1..(k + 1) {
//     //     wind_y[i] = wind_y[i - 1] + dy_out[i - 1];
//     // }
//     //
//     // for i in (k + 1)..dy_out.len() {
//     //     wind_y[i] = wind_y[i - 1] + dy_out[i - 1] - dy_out[i - k - 1];
//     // }
//     //
//     // for i in 1..wind_y.len() {
//     //     if t_out[i] < avg_fac {
//     //         wind_y2[i] = wind_y[i] / t_out[i];
//     //     } else {
//     //         wind_y2[i] = wind_y[i] / avg_fac;
//     //     }
//     // }
//     //
//     // for i in 0..wind_y.len() {
//     //     if t_out[i] < tf {
//     //         wind_y[i] = 0.
//     //     } else if t_out[i] < avg_fac {
//     //         wind_y[i] = wind_y[i] / t_out[i];
//     //     } else {
//     //         wind_y[i] = wind_y[i] / avg_fac;
//     //     }
//     // }
//
//     // let mut smooth_wind_y = vec![0.; dy_out.len()];
//     // smooth_wind_y[0] = wind_y[0];
//     // let mut bt = wind_y[1] - wind_y[0];
//     // for i in 1..dy_out.len() {
//     //     smooth_wind_y[i] = a * wind_y[i] + (1. - a) * (smooth_wind_y[i - 1] + bt);
//     //     bt = b * (smooth_wind_y[i] - smooth_wind_y[i - 1]) + (1. - b) * bt;
//     // }
//
//     // let mut diff_y1 = vec![0.; dy_out.len()];
//     // for i in 0..dy_out.len() {
//     //     diff_y1[i] = bar_y[i] - wind_y[i];
//     // }
//     // let mut diff_y2 = vec![0.; dy_out.len()];
//     // for i in 0..dy_out.len() {
//     //     diff_y2[i] = wind_y[i] - smooth_wind_y[i];
//     // }
//
//     let obsv = rho_out
//         .iter()
//         .map(|rho| fidelity(rho, &rhod))
//         .collect::<Vec<f64>>();
//
//     let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
//         .map(|i| t_out[i * decimation])
//         .collect();
//
//     let obsv_dec = (0..obsv.len() / decimation)
//         .map(|i| obsv[i * decimation])
//         .collect();
//
//     // let mut baryline = plotpy::Curve::new();
//     // baryline.set_line_color(colors[0]).draw(t_out, &bar_y);
//     //
//     // let mut hatyline = plotpy::Curve::new();
//     // hatyline.set_line_color(colors[1]).draw(t_out, &hat_y);
//     //
//     // let mut windyline = plotpy::Curve::new();
//     // windyline.set_line_color(colors[2]).draw(t_out, &wind_y);
//     // let mut windyline2 = plotpy::Curve::new();
//     // windyline2.set_line_color(colors[4]).draw(t_out, &wind_y2);
//
//     // let mut smoothwindyline = plotpy::Curve::new();
//     // smoothwindyline
//     //     .set_line_color(colors[6])
//     //     .draw(t_out, &smooth_wind_y);
//
//     // let mut diffy1line = plotpy::Curve::new();
//     // diffy1line.set_line_color(colors[3]).draw(t_out, &diff_y1);
//     // let mut diffy2line = plotpy::Curve::new();
//     // diffy2line.set_line_color(colors[4]).draw(t_out, &diff_y2);
//
//     let mut fidline = plotpy::Curve::new();
//     fidline
//         .set_line_color(colors[i])
//         .draw(&t_out_dec, &obsv_dec);
//     // let mut yline = plotpy::Curve::new();
//     // yline.set_line_color(colors[5]).draw(t_out, &y);
//
//     // plots[i]
//     // .add(&baryline)
//     // .add(&hatyline)
//     // .add(&windyline2)
//     // .add(&windyline)
//     // .set_subplot(2, 3, 1)
//     // .set_horiz_line(y1 + lb, "#000000", "-", 2.)
//     // .set_horiz_line(y1 + ub, "#00FF00", "-", 2.)
//     // .add(&baryline)
//     // .set_subplot(2, 3, 2)
//     // .set_horiz_line(y1 + lb, "#000000", "-", 2.)
//     // .set_horiz_line(y1 + ub, "#00FF00", "-", 2.)
//     // .add(&hatyline)
//     // .add(&windyline)
//     // .set_subplot(2, 3, 3)
//     // .set_horiz_line(y1 + lb, "#000000", "-", 2.)
//     // .set_horiz_line(y1 + ub, "#00FF00", "-", 2.)
//     // .add(&smoothwindyline)
//     // .set_subplot(2, 3, 4)
//     // .add(&fidline)
//     // .add(&yline)
//     // .set_subplot(2, 3, 5)
//     // .add(&windyline)
//     // .add(&baryline)
//     // .add(&hatyline)
//     // .set_horiz_line(y1 + lb, "#000000", "-", 2.)
//     // .set_horiz_line(y1 + ub, "#00FF00", "-", 2.)
//     // .add(&smoothwindyline)
//     // .add(&diffy1line)
//     // .set_subplot(2, 3, 6)
//     // .add(&windyline)
//     // .add(&windyline2)
//     // .add(&baryline)
//     // .set_horiz_line(y1 + lb, "#000000", "-", 2.)
//     // .set_horiz_line(y1 + ub, "#00FF00", "-", 2.)
//     // .add(&diffy2line);
//     // ;
//     plot.add(&fidline);
// }
// bar.finish();
