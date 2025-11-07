use crate::plots::constrainedlayout;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

/// \[rho_d, L + L^dag\] = 0 case
pub fn timebased() -> SolverResult<()> {
    let mut plot = plotpy::Plot::new();

    let d = 3.;
    let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    let l =
        na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast::<na::Complex<f64>>();
    let f1 = na::Matrix3::zeros();

    let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast::<na::Complex<f64>>();

    // let x0 = na::Vector3::new(1., 1., 1.).cast();
    // let x0 = x0 * x0.conjugate().transpose().scale(1. / 3.);
    let x0 = random_pure_state::<na::U3>();

    let num_tries = 10;
    let final_time: f64 = 10.0;
    let dt = 0.0001;
    let decimation = 10;
    let k = 5000;

    let delta = 3.;
    let gamma = 0.2 * delta;
    let y1 = -2.;
    let epsilon = delta * 1.;
    let beta = 0.2;

    let nu = 0.9;
    let theta = nu * (delta - gamma - l.trace().re.abs() / d - (rhod * l).trace().re.abs());
    let b = GELLMANNMATRICES
        .iter()
        .map(|matrix| (matrix * l).trace().re.powi(2))
        .sum::<f64>()
        .sqrt();
    let lmin = 0.0636;
    let lmax = 1.6340;

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
        let mut rng1 = StdRng::seed_from_u64(5 * i);
        let mut rng2 = StdRng::seed_from_u64(5 * i);
        let mut rng3 = StdRng::seed_from_u64(5 * i);

        let mut system = systems::multilevelcompletefeedback::Feedback::new(
            h,
            l,
            hc,
            na::Matrix3::zeros(),
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
            .map(|rho| 2. * (rho * l).trace().re)
            // .map(|rho| fidelity(rho, &rhod))
            .collect::<Vec<f64>>();

        let mut controlledsystem = systems::timefeedback::Controller::new(
            h, l, hc, f0, f1, y1, k, delta, gamma, beta, epsilon, theta, b, lmin, lmax, &mut rng2,
        );

        let mut controlledsolver =
            StochasticSolver::new(&mut controlledsystem, 0.0, x0, final_time, dt);
        controlledsolver.integrate()?;

        let (ct_out, crho_out, cdy_out) = controlledsolver.results().get();

        let cobsv = crho_out
            .iter()
            .map(|rho| 2. * (rho * l).trace().re)
            // .map(|rho| fidelity(rho, &rhod))
            .collect::<Vec<f64>>();

        let mut controlledsystem2 = systems::multilevelcompletefeedback::Feedback2::new(
            h, l, hc, f0, f1, y1, k, delta, gamma, beta, epsilon, &mut rng3,
        );

        let mut controlledsolver2 =
            StochasticSolver::new(&mut controlledsystem2, 0.0, x0, final_time, dt);
        controlledsolver2.integrate()?;

        let (ct_out2, crho_out2, cdy_out2) = controlledsolver2.results().get();

        let cobsv2 = crho_out2
            .iter()
            .map(|rho| 2. * (rho * l).trace().re)
            // .map(|rho| fidelity(rho, &rhod))
            .collect::<Vec<f64>>();

        let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
            .map(|i| t_out[i * decimation])
            .collect();
        let ct_out_dec: Vec<f64> = (0..ct_out.len() / decimation)
            .map(|i| ct_out[i * decimation])
            .collect();
        let ct_out2_dec: Vec<f64> = (0..ct_out2.len() / decimation)
            .map(|i| ct_out2[i * decimation])
            .collect();

        let obsv_dec = (0..obsv.len() / decimation)
            .map(|i| obsv[i * decimation])
            .collect();
        let cobsv_dec = (0..cobsv.len() / decimation)
            .map(|i| cobsv[i * decimation])
            .collect();
        let cobsv2_dec = (0..cobsv2.len() / decimation)
            .map(|i| cobsv2[i * decimation])
            .collect();

        let mut zaxis = plotpy::Curve::new();
        zaxis
            .set_line_color(colors[i as usize])
            .draw(&t_out_dec, &obsv_dec);

        plot.set_subplot(3, 1, 1).add(&zaxis);

        let mut czaxis = plotpy::Curve::new();
        czaxis
            .set_line_color(colors[i as usize])
            .draw(&ct_out_dec, &cobsv_dec);

        plot.set_subplot(3, 1, 2).add(&czaxis);

        let mut c2zaxis = plotpy::Curve::new();
        c2zaxis
            .set_line_color(colors[i as usize])
            .draw(&ct_out2_dec, &cobsv2_dec);

        plot.set_subplot(3, 1, 3).add(&c2zaxis);
    }

    bar.finish();

    plot.set_subplot(3, 1, 1)
        .set_title(r"$F_0$ inactive")
        .set_label_y("Fidelity");
    plot.set_subplot(3, 1, 2)
        .set_title(r"$F_0$ active with time limit")
        .set_label_y("Fidelity");
    plot.set_subplot(3, 1, 3)
        .set_title(r"$F_0$ active")
        .set_label_y("Fidelity")
        .set_label_x("Time");

    constrainedlayout("Images/multilevelwmreal", &mut plot, true)
}
