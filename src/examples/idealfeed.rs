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

    let l1 = -PAULI_Z + PAULI_X;
    let hc1 = -PAULI_Y;
    let f11 = -PAULI_Y;

    let l2 = -PAULI_Z + PAULI_X.scale(0.1);
    let hc2 = -PAULI_Y.scale(0.1);
    let f12 = -PAULI_Y.scale(0.1);

    let rho1 = na::Matrix2::new(1.0, 0.0, 0.0, 0.0).cast();
    let rho2 = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    let y1 = ((l1 + l1.adjoint()) * rho1).trace().re;
    let y2 = ((l1 + l1.adjoint()) * rho2).trace().re;

    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    // let x0 = random_qubit_state();
    // let x0 = random_pure_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 10.0;
    let dt = 0.0001;
    let decimation = 1;

    let mut avg_sigmaz = vec![0.; (final_time / dt).ceil().to_usize().unwrap() + 1];
    let mut converged_traj = 0;

    let gamma = 0.1;
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

    // let mut rng = StdRng::seed_from_u64(0);
    let mut rng1 = rand::rng();
    let mut rng2 = rand::rng();
    let mut rng3 = rand::rng();
    // let mut ts = vec![0.; avg_sigmaz.len()];

    for i in 0..num_tries {
        bar.inc(1);
        let mut system = systems::idealqubitcompletefeedback::QubitFeedback::new(
            h,
            l1,
            hc1,
            QubitOperator::zeros(),
            f11,
            y1,
            y2,
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

        let mut slowsystem = systems::idealqubitcompletefeedback::QubitFeedback::new(
            h,
            l2,
            hc2,
            QubitOperator::zeros(),
            f12,
            y1,
            y2,
            gamma,
            &mut rng2,
        );
        let mut slowsolver = StochasticSolver::new(&mut slowsystem, 0.0, x0, final_time, dt);
        slowsolver.integrate()?;

        let (st_out, srho_out, sdy_out) = slowsolver.results().get();

        let sobsv = srho_out
            .iter()
            .map(|rho| (PAULI_Z * rho).trace().re)
            .collect::<Vec<f64>>();

        let mut controlledsystem = systems::idealqubitcompletefeedback::QubitFeedback::new(
            h, l2, hc2, f0, f12, y1, y2, gamma, &mut rng3,
        );

        let mut controlledsolver =
            StochasticSolver::new(&mut controlledsystem, 0.0, x0, final_time, dt);
        controlledsolver.integrate()?;

        let (ct_out, crho_out, cdy_out) = controlledsolver.results().get();

        let cobsv = crho_out
            .iter()
            .map(|rho| (PAULI_Z * rho).trace().re)
            .collect::<Vec<f64>>();

        let mut newcorr = vec![0.; crho_out.len()];
        let mut t = 0.;
        for i in 0..cdy_out.len() - 1 {
            t += dt;
            let y = ((l1 + l1.adjoint()) * crho_out[i]).trace().re;
            newcorr[i + 1] = if (y - y2).abs() < ub { 1. } else { 0. };
        }

        if newcorr[newcorr.len() - 1] == 0. {
            converged_traj += 1;
            // avg_sigmaz = avg_sigmaz
            //     .iter()
            //     .zip(&obsv.iter().map(|o| o[2]).collect::<Vec<f64>>())
            //     .map(|(x, y)| x + y)
            //     .collect::<Vec<f64>>();
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

        plot.set_subplot(2, 2, 2).add(&szaxis);

        let mut czaxis = plotpy::Curve::new();
        czaxis
            .set_line_color(colors[i as usize])
            .draw(&ct_out_dec, &cobsv_dec);

        plot.set_subplot(2, 2, 3).add(&czaxis);

        let mut corrs = plotpy::Curve::new();
        corrs
            .set_line_color(colors[i as usize])
            .draw(&ct_out_dec, &newcorr_dec);

        plot.set_subplot(2, 2, 4).add(&corrs);

        // ts = t_out.to_vec();
    }
    bar.finish();

    // let mut sigmaz_avg_traj = plotpy::Curve::new();
    // sigmaz_avg_traj
    //     .set_label(&format!(
    //         "Number of converged trajectories: {converged_traj}"
    //     ))
    //     .draw(
    //         &ts,
    //         &avg_sigmaz
    //             .iter()
    //             .map(|x| x / (converged_traj as f64))
    //             .collect::<Vec<f64>>(),
    //     );
    //
    // plot.add(&sigmaz_avg_traj);

    println!("Converged trajectories: {converged_traj}");
    plot.show("tempimages")?;

    Ok(())
}
