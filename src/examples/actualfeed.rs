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

const NUMTHREADS: usize = 8;

/// \[rho_d, L + L^dag\] = 0 case
pub fn actualfeed() -> SolverResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(NUMTHREADS.min(num_cpus::get()).max(1))
        .build_global()
        .expect("Could not access plot");
    let mut plot = plotpy::Plot::new();

    let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let f1 = na::Matrix3::zeros();

    let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast();

    // let x0 = na::Vector3::new(1., 1., 1.).cast();
    // let x0 = x0 * x0.conjugate().transpose().scale(1. / 3.);

    let num_tries = 50;
    let num_inner_tries = 10;
    let final_time: f64 = 15.0;
    let dt = 0.0001;
    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();
    // let decimation = 60;

    let avg_free_fidelity = Arc::new(Mutex::new(vec![0.; num_steps + 1]));
    let avg_ctrl_fidelity = Arc::new(Mutex::new(vec![0.; num_steps + 1]));
    let avg_time_fidelity = Arc::new(Mutex::new(vec![0.; num_steps + 1]));
    let avg_time_fidelity2 = Arc::new(Mutex::new(vec![0.; num_steps + 1]));
    let avg_ideal_fidelity = Arc::new(Mutex::new(vec![0.; num_steps + 1]));

    let delta = 3.;
    let gamma = 0.2 * delta;
    let y1 = -2.;
    let epsilon = delta * 1.;
    let beta = 0.9;
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
        let x0 = random_pure_state::<na::U3>();

        // for j in 0..num_inner_tries {
        // }
        (0..num_inner_tries)
            .into_par_iter()
            .map(|j| -> Result<(), SolverError> {
                let mut rng1 = StdRng::seed_from_u64(num_inner_tries * i + j);
                let mut rng2 = StdRng::seed_from_u64(num_inner_tries * i + j);
                let mut rng3 = StdRng::seed_from_u64(num_inner_tries * i + j);
                let mut rng4 = StdRng::seed_from_u64(num_inner_tries * i + j);
                let mut rng5 = StdRng::seed_from_u64(num_inner_tries * i + j);
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
                    .map(|rho| fidelity(rho, &rhod))
                    .collect::<Vec<f64>>();

                let mut free_fidelity = avg_free_fidelity
                    .lock()
                    .expect("Could not access free fidelity lock");
                *free_fidelity = free_fidelity
                    .iter()
                    .zip(&obsv)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<f64>>();
                // avg_free_fidelity = avg_free_fidelity
                //     .iter()
                //     .zip(&obsv)
                //     .map(|(x, y)| x + y)
                //     .collect::<Vec<f64>>();

                let mut controlledsystem = systems::multilevelcompletefeedback::Feedback::new(
                    h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon, &mut rng2,
                );
                let mut controlledsolver =
                    StochasticSolver::new(&mut controlledsystem, 0.0, x0, final_time, dt);
                controlledsolver.integrate()?;

                let (st_out, srho_out, sdy_out) = controlledsolver.results().get();

                let sobsv = srho_out
                    .iter()
                    .map(|rho| fidelity(rho, &rhod))
                    .collect::<Vec<f64>>();

                let mut ctrl_fidelity = avg_ctrl_fidelity
                    .lock()
                    .expect("Could not access ctrl fidelity lock");
                *ctrl_fidelity = ctrl_fidelity
                    .iter()
                    .zip(&sobsv)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<f64>>();
                // avg_ctrl_fidelity = avg_ctrl_fidelity
                //     .iter()
                //     .zip(&sobsv)
                //     .map(|(x, y)| x + y)
                //     .collect::<Vec<f64>>();

                let mut timecontrolledsystem = systems::multilevelcompletefeedback::Feedback2::new(
                    h, l, hc, f0, f1, y1, k, delta, gamma, beta, epsilon, &mut rng3,
                );
                let mut timecontrolledsolver =
                    StochasticSolver::new(&mut timecontrolledsystem, 0.0, x0, final_time, dt);
                timecontrolledsolver.integrate()?;

                let (tt_out, trho_out, tdy_out) = timecontrolledsolver.results().get();

                let tobsv = trho_out
                    .iter()
                    .map(|rho| fidelity(rho, &rhod))
                    .collect::<Vec<f64>>();

                let mut time_fidelity = avg_time_fidelity
                    .lock()
                    .expect("Could not access time fidelity lock");
                *time_fidelity = time_fidelity
                    .iter()
                    .zip(&tobsv)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<f64>>();
                // avg_time_fidelity = avg_time_fidelity
                //     .iter()
                //     .zip(&tobsv)
                //     .map(|(x, y)| x + y)
                //     .collect::<Vec<f64>>();

                let mut idealcontrolledsystem =
                    systems::idealmultilevelcompletefeedback::Feedback::new(
                        h, l, hc, f0, f1, y1, delta, gamma, &mut rng4,
                    );

                let mut idealcontrolledsolver =
                    StochasticSolver::new(&mut idealcontrolledsystem, 0.0, x0, final_time, dt);
                idealcontrolledsolver.integrate()?;

                let (it_out, irho_out, idy_out) = idealcontrolledsolver.results().get();

                let iobsv = irho_out
                    .iter()
                    .map(|rho| fidelity(rho, &rhod))
                    .collect::<Vec<f64>>();

                let mut ideal_fidelity = avg_ideal_fidelity
                    .lock()
                    .expect("Could not access ideal fidelity lock");
                *ideal_fidelity = ideal_fidelity
                    .iter()
                    .zip(&iobsv)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<f64>>();
                // avg_ideal_fidelity = avg_ideal_fidelity
                //     .iter()
                //     .zip(&iobsv)
                //     .map(|(x, y)| x + y)
                //     .collect::<Vec<f64>>();

                let mut timecontrolledsystem2 = systems::multilevelcompletefeedback::Feedback2::new(
                    h, l, hc, f0, f1, y1, k2, delta, gamma, beta, epsilon, &mut rng3,
                );
                let mut timecontrolledsolver2 =
                    StochasticSolver::new(&mut timecontrolledsystem2, 0.0, x0, final_time, dt);
                timecontrolledsolver2.integrate()?;

                let (tt_out2, trho_out2, tdy_out2) = timecontrolledsolver2.results().get();

                let tobsv2 = trho_out2
                    .iter()
                    .map(|rho| fidelity(rho, &rhod))
                    .collect::<Vec<f64>>();

                let mut time2_fidelity = avg_time_fidelity2
                    .lock()
                    .expect("Could not access time2 fidelity lock");
                *time2_fidelity = time2_fidelity
                    .iter()
                    .zip(&tobsv2)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<f64>>();
                // avg_time_fidelity2 = avg_time_fidelity2
                //     .iter()
                //     .zip(&tobsv2)
                //     .map(|(x, y)| x + y)
                //     .collect::<Vec<f64>>();
                Ok(())
            })
            .collect::<Result<Vec<()>, SolverError>>()?;

        // let t_out_dec: Vec<f64> = (0..t_out.len() / decimation)
        //     .map(|i| t_out[i * decimation])
        //     .collect();
        // let st_out_dec: Vec<f64> = (0..st_out.len() / decimation)
        //     .map(|i| st_out[i * decimation])
        //     .collect();
        //
        // let obsv_dec = (0..obsv.len() / decimation)
        //     .map(|i| obsv[i * decimation])
        //     .collect();
        // let sobsv_dec = (0..sobsv.len() / decimation)
        //     .map(|i| sobsv[i * decimation])
        //     .collect();
        //
        // let mut zaxis = plotpy::Curve::new();
        // zaxis
        //     .set_line_color(colors[i as usize])
        //     .draw(&t_out_dec, &obsv_dec);
        //
        // plot.set_subplot(2, 1, 1).add(&zaxis);
        //
        // let mut szaxis = plotpy::Curve::new();
        // szaxis
        //     .set_line_color(colors[i as usize])
        //     .draw(&st_out_dec, &sobsv_dec);
        //
        // plot.set_subplot(2, 1, 2).add(&szaxis);
    }
    bar.finish();

    let t_out = (0..=num_steps)
        .map(|n| (n as f64) * dt)
        .collect::<Vec<f64>>();

    let avg_free_fidelity = avg_free_fidelity
        .lock()
        .expect("Could not take free lock while plotting")
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_ctrl_fidelity = avg_ctrl_fidelity
        .lock()
        .expect("Could not take ctrl lock while plotting")
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_time_fidelity = avg_time_fidelity
        .lock()
        .expect("Could not take time lock while plotting")
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_ideal_fidelity = avg_ideal_fidelity
        .lock()
        .expect("Could not take ideal lock while plotting")
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_time_fidelity2 = avg_time_fidelity2
        .lock()
        .expect("Could not take time2 lock while plotting")
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let mut free_curve = plotpy::Curve::new();
    free_curve
        .set_label("Free evolution")
        .draw(&t_out, &avg_free_fidelity);

    let mut ctrl_curve = plotpy::Curve::new();
    ctrl_curve
        .set_label("Controlled evolution")
        .draw(&t_out, &avg_ctrl_fidelity);

    let mut time_curve = plotpy::Curve::new();
    time_curve
        .set_label(&format!("Windowed evolution, k = {}", k))
        .draw(&t_out, &avg_time_fidelity);

    let mut ideal_curve = plotpy::Curve::new();
    ideal_curve
        .set_label("Ideal evolution")
        .draw(&t_out, &avg_ideal_fidelity);

    let mut time_curve2 = plotpy::Curve::new();
    time_curve2
        .set_label(&format!("Windowed evolution, k = {}", k2))
        .draw(&t_out, &avg_time_fidelity2);

    plot.add(&free_curve)
        .add(&ctrl_curve)
        .add(&time_curve)
        .add(&ideal_curve)
        .add(&time_curve2)
        .legend();

    constrainedlayout("Images/multilevelwmreal", &mut plot, true)
}
