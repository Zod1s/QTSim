use crate::plots::constrainedlayout;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// \[rho_d, L + L^dag\] = 0 case
pub fn actualfeed() -> SolverResult<()> {
    let mut plot = plotpy::Plot::new();

    // let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    // let hc = na::Matrix3::zeros();
    // let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    // let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    // let f1 = na::Matrix3::zeros();
    //
    // let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast();

    let h = ferromagnetictriangle(&vec![1., 1., 5.]);
    let l = h.clone();
    let hc = Operator::<na::U8>::zeros();
    let f1 = hc.clone();
    let f0 = Operator::<na::U8>::from_element(na::Complex::ONE) - Operator::<na::U8>::identity();

    let num_tries = 10;
    let final_time: f64 = 10.0;
    let dt = 0.0001;
    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();
    let decimation = 1;

    let mut avg_free_fidelity = vec![0.; num_steps + 1];
    let mut avg_ctrl_fidelity = vec![0.; num_steps + 1];
    let mut avg_time_fidelity = vec![0.; num_steps + 1];
    let mut avg_time_fidelity2 = vec![0.; num_steps + 1];
    let mut avg_ideal_fidelity = vec![0.; num_steps + 1];

    let delta = 3.;
    let gamma = 0.2 * delta;
    let y1 = -2.;
    let epsilon = delta * 1.;
    let beta = 0.6;
    let k = 5000;
    let k2 = 10000;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for i in 0..num_tries {
        bar.inc(1);
        let x0 = random_pure_state::<na::U8>(Some(10 * (i + 1)));

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
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            // .map(|rho| fidelity(rho, &rhod))
            .map(|rho| {
                (rho * (Operator::from_diagonal(
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

        let mut controlledsystem = systems::multilevelcompletefeedback::Feedback::new(
            h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon, &mut rng2,
        );
        let mut controlledsolver =
            StochasticSolver::new(&mut controlledsystem, 0.0, x0, final_time, dt);
        controlledsolver.integrate()?;

        let (st_out, srho_out, sdy_out) = controlledsolver.results().get();

        let sobsv = srho_out
            .iter()
            // .map(|rho| fidelity(rho, &rhod))
            .map(|rho| {
                (rho * (Operator::from_diagonal(
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

        let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
            .map(|i| t_out[i * decimation])
            .collect();
        let st_out_dec: Vec<f64> = (0..st_out.len() / decimation)
            .map(|i| st_out[i * decimation])
            .collect();

        let obsv_dec = (0..obsv.len() / decimation)
            .map(|i| obsv[i * decimation])
            .collect();
        let meas_dec = (0..meas.len() / decimation)
            .map(|i| meas[i * decimation])
            .collect();
        let sobsv_dec = (0..sobsv.len() / decimation)
            .map(|i| sobsv[i * decimation])
            .collect();
        let smeas_dec = (0..meas.len() / decimation)
            .map(|i| meas[i * decimation])
            .collect();

        let mut zaxis = plotpy::Curve::new();
        zaxis
            .set_line_color(colors[i as usize])
            .draw(&t_out_dec, &obsv_dec);

        plot.set_subplot(2, 2, 1).add(&zaxis);

        let mut maxis = plotpy::Curve::new();
        maxis
            .set_line_color(colors[i as usize])
            .draw(&t_out_dec, &meas_dec);

        plot.set_subplot(2, 2, 2).add(&maxis);

        let mut szaxis = plotpy::Curve::new();
        szaxis
            .set_line_color(colors[i as usize])
            .draw(&st_out_dec, &sobsv_dec);

        plot.set_subplot(2, 2, 3).add(&szaxis);

        let mut smaxis = plotpy::Curve::new();
        smaxis
            .set_line_color(colors[i as usize])
            .draw(&st_out_dec, &smeas_dec);

        plot.set_subplot(2, 2, 4).add(&smaxis);
    }
    bar.finish();

    constrainedlayout("Images/multilevelwmreal1", &mut plot, true)
}

fn ferromagnetictriangle(weights: &[f64]) -> Operator<na::U8> {
    let id = Operator::<na::U2>::identity();
    let s1 = PAULIS
        .iter()
        .map(|pauli| pauli.kronecker(&id).kronecker(&id))
        .collect::<Vec<Operator<na::U8>>>();

    let s2 = PAULIS
        .iter()
        .map(|pauli| id.kronecker(&pauli).kronecker(&id))
        .collect::<Vec<Operator<na::U8>>>();

    let s3 = PAULIS
        .iter()
        .map(|pauli| id.kronecker(&id).kronecker(&pauli))
        .collect::<Vec<Operator<na::U8>>>();

    let h = -s1
        .iter()
        .zip(s2.iter())
        .zip(weights.iter())
        .map(|((p1, p2), w)| p1 * p2.scale(*w))
        .sum::<Operator<na::U8>>()
        - s2.iter()
            .zip(s3.iter())
            .zip(weights.iter())
            .map(|((p1, p2), w)| p1 * p2.scale(*w))
            .sum::<Operator<na::U8>>()
        - s3.iter()
            .zip(s1.iter())
            .zip(weights.iter())
            .map(|((p1, p2), w)| p1 * p2.scale(*w))
            .sum::<Operator<na::U8>>();

    h
}
