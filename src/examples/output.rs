use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const NUMTHREADS: usize = 8;

pub fn output() -> SolverResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(NUMTHREADS.min(num_cpus::get()).max(1))
        .build_global()
        .expect("Could not build threadpool");

    let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    // let hc = na::Matrix3::zeros();
    // let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let f1 = na::Matrix3::zeros();

    let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast();
    let x0 = na::Vector3::new(1., 1., 1.).cast();
    let x0 = x0 * x0.conjugate().transpose().scale(1. / 3.);

    let num_tries = 10;
    let final_time: f64 = 20.0;
    let dt = 0.0001;
    let decimation = 20;

    let k = 20000;
    let avg_fac = (k as f64) * dt;
    let eta = 0.01;

    let delta = 3.;
    let gamma = 0.2 * delta;
    let y1 = -2.;
    let epsilon = delta * 1.;
    let beta = 0.9;

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

    plot.lock()
        .unwrap()
        .extra("plt.rcParams.update({\"text.usetex\": True, \"font.family\": \"Helvetica\"})\n")
        .extra("plt.rcParams['figure.constrained_layout.use'] = True\n")
        .set_figure_size_inches(7., 5.)
        .set_save_tight(true);

    for i in 0..num_tries {
        bar.inc(1);
        let mut system = systems::sse::SSE::new(
            h,
            l,
            // hc,
            // f0,
            // f1,
            eta,
            // y1,
            // k,
            // delta,
            // gamma,
            // beta,
            // epsilon,
            Some(i),
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
            .set_line_color(colors[i as usize])
            .draw(&t_out_dec, &obsv_dec);

        plot.lock().expect("Could not access plot").add(&fidline);
    }
    bar.finish();

    let mut system = systems::wisemanfme::WisemanFME::new(h, l, f1);
    let mut solver = Rk4::new(system, 0.0, x0, 20.0, 0.001);
    solver.integrate();

    let (t_out, rho_out) = solver.results().get();

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
        .set_line_color("#000000")
        .draw(&t_out_dec, &obsv_dec);

    plot.lock().expect("Could not access plot").add(&fidline);

    plot.lock()
        .expect("Could not access plot")
        // .legend()
        .show("tempimages")?;

    Ok(())
}
