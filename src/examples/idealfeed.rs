use crate::plots::constrainedlayout;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::{ConstOne, ToPrimitive};

/// \[rho_d, L + L^dag\] = 0 case
pub fn idealfeed() -> SolverResult<()> {
    let mut plot = plotpy::Plot::new();

    let h = PAULI_Z;
    let l = PAULI_Z;
    let hc = QubitOperator::zeros();
    let f0 = PAULI_X.scale(0.1);
    let f1 = QubitOperator::zeros();

    // let h = PAULI_Z;
    // let f0 = PAULI_X;
    //
    // let l1 = -PAULI_Z + PAULI_X.scale(1.0);
    // let hc1 = -PAULI_Y.scale(1.0);
    // let f11 = -PAULI_Y.scale(1.0);
    //
    // let l2 = -PAULI_Z + PAULI_X.scale(0.1);
    // let hc2 = -PAULI_Y.scale(0.1);
    // let f12 = -PAULI_Y.scale(0.1);

    // let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    // let hc = na::Matrix3::zeros();
    // let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.]
    //     .cast()
    //     .scale(0.1);
    // let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    // let f1 = na::Matrix3::zeros();

    // let a: f64 = 0.05;
    // let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1., 2., 3.)).cast();
    // let hc = na::matrix![na::Complex::ONE, na::Complex::new(0., 0.5 * a), -na::Complex::new(0., a.powi(2)); na::Complex::new(0., -0.5 * a), na::Complex::ONE.scale(-2.), na::Complex::new(0., -2.5 * a); na::Complex::new(0., a.powi(2)), na::Complex::new(0., 2.5 * a), na::Complex::new(-3., 0.)];
    // let l = na::matrix![-1., a, 0.; a, 2., a; 0., a, 3.].cast();
    // let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    // let f1 = na::matrix![0., a, 0.; -a, 0., a; 0., -a, 0.].cast() * na::Complex::I;

    let rho1 = na::Matrix2::new(1.0, 0.0, 0.0, 0.0).cast();
    let rho2 = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    let y1 = ((l + l.adjoint()) * rho1).trace().re;
    let y2 = ((l + l.adjoint()) * rho2).trace().re;
    // let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast();

    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    // let x0 = random_unit_complex_vector::<3>();
    // let x0 = na::Vector3::new(1., 1., 1.).cast();
    // let x0 = x0 * x0.conjugate().transpose().scale(1. / 3.);
    // let x0 = random_qubit_state();
    // let x0 = random_pure_state();
    // let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 100.0;
    let dt = 0.0001;
    let decimation = 100;

    // let mut avg_sigmaz = vec![0.; (final_time / dt).ceil().to_usize().unwrap() + 1];
    // let mut converged_traj = 0;

    // let delta = 3.;
    // let gamma = 0.4 * delta;
    // let y1 = -2.;
    let gamma = 0.4;
    // let ub = (y1 - y2).abs() * gamma;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    // let mut rng1 = rand::rng();
    // let mut rng2 = rand::rng();
    let mut rng1 = StdRng::seed_from_u64(0);
    // let mut rng3 = rand::rng();
    let mut rng2 = StdRng::seed_from_u64(0);
    // let mut rng3 = StdRng::seed_from_u64(0);

    for i in 0..num_tries {
        bar.inc(1);
        let mut system = systems::idealqubitcompletefeedback::QubitFeedback::new(
            h,
            l,
            hc,
            QubitOperator::zeros(),
            f1,
            y1,
            y2,
            gamma,
            &mut rng1,
        );
        // let mut system = systems::idealmultilevelcompletefeedback::Feedback::new(
        //     h,
        //     l,
        //     hc,
        //     na::Matrix3::zeros(),
        //     f1,
        //     y1,
        //     delta,
        //     gamma,
        //     &mut rng1,
        // );

        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate();

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| fidelity(rho, &rho1))
            // .map(|rho| (l * rho).trace().re)
            .collect::<Vec<f64>>();

        let mut controlledsystem = systems::idealqubitcompletefeedback::QubitFeedback::new(
            h, l, hc, f0, f1, y1, y2, gamma, &mut rng2,
        );
        // let mut controlledsystem = systems::idealmultilevelcompletefeedback::Feedback::new(
        //     h, l, hc, f0, f1, y1, delta, gamma, &mut rng2,
        // );
        let mut controlledsolver =
            StochasticSolver::new(&mut controlledsystem, 0.0, x0, final_time, dt);
        controlledsolver.integrate();

        let (st_out, srho_out, sdy_out) = controlledsolver.results().get();

        let sobsv = srho_out
            .iter()
            .map(|rho| fidelity(rho, &rho1))
            .collect::<Vec<f64>>();

        // let mut controlledsystem2 = systems::idealmultilevelcompletefeedback::Feedback::new(
        //     h, l2, hc2, f0, f12, y1, y2, gamma, &mut rng3,
        // );
        //
        // let mut controlledsolver2 =
        //     StochasticSolver::new(&mut controlledsystem2, 0.0, x0, final_time, dt);
        // controlledsolver2.integrate();
        //
        // let (ct_out, crho_out, cdy_out) = controlledsolver2.results().get();
        //
        // let cobsv = crho_out
        //     .iter()
        //     .map(|rho| fidelity(rho, &rho1))
        //     .collect::<Vec<f64>>();

        let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
            .map(|i| t_out[i * decimation])
            .collect();
        let st_out_dec: Vec<f64> = (0..st_out.len() / decimation)
            .map(|i| st_out[i * decimation])
            .collect();
        // let ct_out_dec: Vec<f64> = (0..ct_out.len() / decimation)
        //     .map(|i| ct_out[i * decimation])
        //     .collect();

        let obsv_dec = (0..obsv.len() / decimation)
            .map(|i| obsv[i * decimation])
            .collect();
        let sobsv_dec = (0..sobsv.len() / decimation)
            .map(|i| sobsv[i * decimation])
            .collect();
        // let cobsv_dec = (0..cobsv.len() / decimation)
        //     .map(|i| cobsv[i * decimation])
        //     .collect();

        let mut zaxis = plotpy::Curve::new();
        zaxis
            .set_line_color(colors[i as usize])
            .draw(&t_out_dec, &obsv_dec);

        plot.set_subplot(2, 1, 1).add(&zaxis);

        let mut szaxis = plotpy::Curve::new();
        szaxis
            .set_line_color(colors[i as usize])
            .draw(&st_out_dec, &sobsv_dec);

        plot.set_subplot(2, 1, 2).add(&szaxis);
        // plot.add(&zaxis);

        // let mut czaxis = plotpy::Curve::new();
        // czaxis
        //     .set_line_color(colors[i as usize])
        //     .draw(&ct_out_dec, &cobsv_dec);
        //
        // plot.set_subplot(3, 1, 3).add(&czaxis);

        println!("No F0: {}", obsv[obsv.len() - 1]);
        println!("F0: {}", sobsv[sobsv.len() - 1]);
    }

    bar.finish();

    // plot.set_subplot(2, 1, 1)
    //     .set_title(r"$F_0$ inactive")
    //     .set_label_y("Fidelity");
    // plot.set_subplot(3, 1, 2)
    //     .set_title(r"Second system with $F_0 = 0$")
    //     .set_label_y("Fidelity");
    // plot.set_subplot(2, 1, 2)
    //     .set_title(r"$F_0$ active")
    //     .set_label_y("Fidelity")
    //     .set_label_x("Time");

    constrainedlayout("Images/temp", &mut plot, true)
}
