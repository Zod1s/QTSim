use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

/// \[rho_d, L + L^dag\] = 0 case
pub fn qnd() -> SolverResult<()> {
    let mut plot = plotpy::Plot::new();

    let h = -PAULI_Z;
    let f0 = PAULI_X;

    // let l1 = -PAULI_Z + PAULI_X;
    // let hc1 = -PAULI_Y;
    // let f11 = -PAULI_Y;

    let l = -PAULI_Z + PAULI_X.scale(0.1);
    let hc = -PAULI_Y.scale(0.1);
    let f1 = -PAULI_Y.scale(0.1);

    // let h = PAULI_Z;
    // let hc = QubitOperator::zeros();
    // let l = -PAULI_Z;
    // let f0 = PAULI_X;
    // let f1 = QubitOperator::zeros();

    let rho1 = na::Matrix2::new(1.0, 0.0, 0.0, 0.0).cast();
    let rho2 = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    let y1 = ((l + l.adjoint()) * rho1).trace().re;
    let y2 = ((l + l.adjoint()) * rho2).trace().re;

    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    // let x0 = random_qubit_state();
    // let x0 = random_pure_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 10.0;
    let dt = 0.00001;
    let decimation = 10;

    let mut avg_sigmaz = vec![0.; (final_time / dt).ceil().to_usize().unwrap() + 1];
    let mut converged_traj = 0;

    let gamma = 0.4;
    let ub = (y1 - y2).abs() * gamma;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    let mut rng1 = rand::rng();
    // let mut rng2 = rand::rng();
    let mut rng2 = StdRng::seed_from_u64(0);
    // let mut rng3 = rand::rng();
    let mut rng3 = StdRng::seed_from_u64(0);

    for i in 0..num_tries {
        bar.inc(1);
        let mut system = systems::idealmultilevelcompletefeedback::Feedback::new(
            h,
            l,
            hc,
            QubitOperator::zeros(),
            f1,
            y1,
            (y2 - y1) / 2.,
            gamma,
            &mut rng1,
        );

        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| (PAULI_Z * rho).trace().re)
            .collect::<Vec<f64>>();

        let mut controlledsystem = systems::idealmultilevelcompletefeedback::Feedback::new(
            h,
            l,
            hc,
            f0,
            f1,
            y1,
            (y2 - y1) / 2.,
            gamma,
            &mut rng2,
        );
        let mut controlledsolver =
            StochasticSolver::new(&mut controlledsystem, 0.0, x0, final_time, dt);
        controlledsolver.integrate()?;

        let (st_out, srho_out, sdy_out) = controlledsolver.results().get();

        let sobsv = srho_out
            .iter()
            .map(|rho| (PAULI_Z * rho).trace().re)
            .collect::<Vec<f64>>();

        let mut controlledsystem2 = systems::idealmultilevelcompletefeedback::Feedback::new(
            h,
            l,
            hc,
            f0,
            f1,
            y1,
            2.,
            gamma / 2.,
            &mut rng3,
        );

        let mut controlledsolver2 =
            StochasticSolver::new(&mut controlledsystem2, 0.0, x0, final_time, dt);
        controlledsolver2.integrate()?;

        let (ct_out, crho_out, cdy_out) = controlledsolver2.results().get();

        let cobsv = crho_out
            .iter()
            .map(|rho| (PAULI_Z * rho).trace().re)
            .collect::<Vec<f64>>();

        let mut newcorr = vec![0.; crho_out.len()];
        let mut t = 0.;
        for i in 0..cdy_out.len() - 1 {
            t += dt;
            let y = ((l + l.adjoint()) * crho_out[i]).trace().re;
            newcorr[i + 1] = if (y - y2).abs() < ub { 1. } else { 0. };
        }

        let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
            .map(|i| t_out[i * decimation])
            .collect();
        let st_out_dec: Vec<f64> = (0..st_out.len() / decimation)
            .map(|i| st_out[i * decimation])
            .collect();
        let ct_out_dec: Vec<f64> = (0..ct_out.len() / decimation)
            .map(|i| ct_out[i * decimation])
            .collect();

        let obsv_dec = (0..obsv.len() / decimation)
            .map(|i| obsv[i * decimation])
            .collect();
        let sobsv_dec = (0..sobsv.len() / decimation)
            .map(|i| sobsv[i * decimation])
            .collect();
        let cobsv_dec = (0..cobsv.len() / decimation)
            .map(|i| cobsv[i * decimation])
            .collect();
        let newcorr_dec = (0..newcorr.len() / decimation)
            .map(|i| newcorr[i * decimation])
            .collect();

        let mut zaxis = plotpy::Curve::new();
        zaxis
            .set_line_color(colors[i as usize])
            .draw(&t_out_dec, &obsv_dec);

        plot.set_subplot(2, 2, 1).add(&zaxis);

        let mut szaxis = plotpy::Curve::new();
        szaxis
            .set_line_color(colors[i as usize])
            .draw(&st_out_dec, &sobsv_dec);

        plot.set_subplot(2, 2, 3).add(&szaxis);

        let mut czaxis = plotpy::Curve::new();
        czaxis
            .set_line_color(colors[i as usize])
            .draw(&ct_out_dec, &cobsv_dec);

        plot.set_subplot(2, 2, 2).add(&czaxis);

        let mut corrs = plotpy::Curve::new();
        corrs
            .set_line_color(colors[i as usize])
            .draw(&ct_out_dec, &newcorr_dec);

        plot.set_subplot(2, 2, 4).add(&corrs);
    }
    bar.finish();

    plot.show("tempimages")?;

    Ok(())
}
