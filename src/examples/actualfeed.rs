use crate::plots::constrainedlayout;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use std::fs::File;
use std::sync::{Arc, Mutex};

/// \[rho_d, L + L^dag\] = 0 case
pub fn actualfeed() -> SolverResult<()> {
    // let mut plot = plotpy::Plot::new();

    let h = ferromagnetictriangle(&vec![0.5, 0.5, 3.0]);
    let l = h.clone();
    let hc = crate::utils::Operator::<na::U8>::zeros();
    let f1 = hc.clone();

    let eigen = h.symmetric_eigen();
    let f0 = eigen.eigenvectors
        * crate::utils::Operator::<na::U8>::from_fn(|i, j| {
            if i == j - 1 || i == j + 1 {
                na::Complex::ONE
            } else {
                na::Complex::ZERO
            }
        })
        .scale(6.)
        * eigen.eigenvectors.adjoint();

    let num_tries = 20;
    let final_time: f64 = 10.0;
    let dt = 0.0001;
    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();
    let decimation = 10;

    let mut avg_u_fidelity = vec![0.; (num_steps + 1) / decimation];
    let mut avg_c_fidelity = vec![0.; (num_steps + 1) / decimation];

    let delta = 10.;
    let gamma = 0.2 * delta;
    let y1 = 2. * eigen.eigenvalues.min();
    let epsilon = delta * 1.;
    let beta = 0.6;
    let k = 5000;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    let x0 = random_pure_state::<na::U8>(Some(1000));

    for i in 0..num_tries {
        bar.inc(1);

        let mut rng1 = StdRng::seed_from_u64(i);
        let mut rng2 = StdRng::seed_from_u64(i);
        let mut system = systems::multilevelcompletefeedback::Feedback::new(
            h,
            l,
            hc,
            hc.clone(),
            f1,
            y1,
            delta,
            gamma,
            beta,
            epsilon,
            &mut rng1,
        );

        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate();

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| {
                (rho * (crate::utils::Operator::from_diagonal(
                    &na::vector![1., 0., 0., 0., 0., 0., 0., 1.].cast(),
                )))
                .trace()
                .re
            })
            .collect::<Vec<f64>>();

        let meas = rho_out
            .iter()
            .map(|rho| (rho * l).trace().re)
            .collect::<Vec<f64>>();

        // let mut controlledsystem = systems::multilevelcompletefeedback::Feedback::new(
        //     h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon, &mut rng2,
        // );
        // let mut controlledsystem = systems::multilevelcompletefeedback::Feedback2::new(
        //     h, l, hc, f0, f1, y1, k, delta, gamma, beta, epsilon, &mut rng2,
        // );
        let mut controlledsystem = systems::idealmultilevelcompletefeedback::Feedback::new(
            h, l, hc, f0, f1, y1, delta, gamma, &mut rng2,
        );
        let mut controlledsolver =
            StochasticSolver::new(&mut controlledsystem, 0.0, x0, final_time, dt);
        controlledsolver.integrate();

        let (st_out, srho_out, sdy_out) = controlledsolver.results().get();

        let sobsv = srho_out
            .iter()
            .map(|rho| {
                (rho * (crate::utils::Operator::from_diagonal(
                    &na::vector![1., 0., 0., 0., 0., 0., 0., 1.].cast(),
                )))
                .trace()
                .re
            })
            .collect::<Vec<f64>>();
        let smeas = srho_out
            .iter()
            .map(|rho| (rho * l).trace().re)
            .collect::<Vec<f64>>();

        // let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
        //     .map(|i| t_out[i * decimation])
        //     .collect();
        // let st_out_dec: Vec<f64> = (0..st_out.len() / decimation)
        //     .map(|i| st_out[i * decimation])
        //     .collect();

        let obsv_dec = (0..obsv.len() / decimation)
            .map(|i| obsv[i * decimation])
            .collect::<Vec<f64>>();
        // let meas_dec = (0..meas.len() / decimation)
        //     .map(|i| meas[i * decimation])
        //     .collect::<Vec<f64>>();

        let sobsv_dec = (0..sobsv.len() / decimation)
            .map(|i| sobsv[i * decimation])
            .collect::<Vec<f64>>();
        // let smeas_dec = (0..smeas.len() / decimation)
        //     .map(|i| smeas[i * decimation])
        //     .collect::<Vec<f64>>();

        // let mut zaxis = plotpy::Curve::new();
        // zaxis.draw(&t_out_dec, &obsv_dec);
        //
        // plot.set_subplot(2, 2, 1).add(&zaxis);
        //
        // let mut maxis = plotpy::Curve::new();
        // maxis.draw(&t_out_dec, &meas_dec);
        //
        // plot.set_subplot(2, 2, 3).add(&maxis);

        avg_u_fidelity = avg_u_fidelity
            .iter()
            .zip(&obsv)
            .map(|(x, y)| x + y)
            .collect::<Vec<f64>>();

        avg_c_fidelity = avg_c_fidelity
            .iter()
            .zip(&sobsv)
            .map(|(x, y)| x + y)
            .collect::<Vec<f64>>();

        // let mut szaxis = plotpy::Curve::new();
        // szaxis.draw(&st_out_dec, &sobsv_dec);
        //
        // plot.set_subplot(2, 2, 2).add(&szaxis);
        //
        // let mut smaxis = plotpy::Curve::new();
        // smaxis.draw(&st_out_dec, &smeas_dec);
        //
        // plot.set_subplot(2, 2, 4).add(&smaxis);
    }
    bar.finish();

    avg_u_fidelity = avg_u_fidelity
        .iter()
        .map(|f| f / (num_tries as f64))
        .collect::<Vec<f64>>();

    avg_c_fidelity = avg_c_fidelity
        .iter()
        .map(|f| f / (num_tries as f64))
        .collect::<Vec<f64>>();

    let t_out = (0..=num_steps)
        .map(|n| (n as f64) * dt)
        .collect::<Vec<f64>>();

    let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
        .map(|i| t_out[i * decimation])
        .collect();

    // println!("Plotting");
    // constrainedlayout("Images/multilevelwmreal", &mut plot, true)

    println!("Saving data");

    let mut file = File::create("./save.csv").expect("Could not create file");

    let mut df: DataFrame = df!(
        "time" => t_out_dec,
        "avg_u" => avg_u_fidelity,
        "avg_c" => avg_c_fidelity,
    )
    .unwrap();

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df);

    Ok(())
}
