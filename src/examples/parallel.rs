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

const NUMTHREADS: usize = 30;

pub fn parallel() -> SolverResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(NUMTHREADS.min(num_cpus::get()).max(1))
        .build_global()
        .expect("Could not create threadpool");
    let mut plot = plotpy::Plot::new();

    // let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    // let hc = na::Matrix3::zeros();
    // let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    // let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    // let f1 = na::Matrix3::zeros();
    //
    // let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast();
    let h = ferromagnetictriangle(&vec![0.5, 0.5, 3.0]);
    let l = h.clone();
    let hc = Operator::<na::U8>::zeros();
    let f1 = hc.clone();

    let eigen = h.symmetric_eigen();
    let f0 = eigen.eigenvectors
        * Operator::<na::U8>::from_fn(|i, j| {
            if i == j - 1 || i == j + 1 {
                na::Complex::ONE
            } else {
                na::Complex::ZERO
            }
        })
        .scale(4.)
        * eigen.eigenvectors.adjoint();
    let rhod = Operator::from_diagonal(&na::vector![1., 0., 0., 0., 0., 0., 0., 1.].cast());

    let num_tries = 500;
    let num_inner_tries = 10;
    let final_time: f64 = 10.0;
    let dt = 0.0001;
    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();

    let mut avg_free_fidelity = vec![0.; num_steps + 1];
    let mut avg_ctrl_fidelity = vec![0.; num_steps + 1];
    let mut avg_time_fidelity1 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity2 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity3 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity4 = vec![0.; num_steps + 1];
    let mut avg_ideal_fidelity = vec![0.; num_steps + 1];

    let delta = 10.;
    let gamma = 0.2 * delta;
    let y1 = 2. * eigen.eigenvalues.min();
    let epsilon = delta * 1.;
    let beta = 0.6;
    let k1 = 5000;
    let k2 = 20000;
    let k3 = 50000;
    let k4 = 100000;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(7 * num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    rayon::scope(|s| {
        s.spawn(|s| {
            for i in 0..num_tries {
                bar.inc(1);
                let x0 = random_pure_state::<na::U8>(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback::new(
                        h,
                        l,
                        hc,
                        Operator::<na::U8>::zeros(),
                        f1,
                        y1,
                        delta,
                        gamma,
                        beta,
                        epsilon,
                        &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();
                    let (t_out, rho_out, dy_out) = solver.results().get();

                    let obsv = rho_out
                        .iter()
                        .map(|rho| fidelity(rho, &rhod))
                        .collect::<Vec<f64>>();

                    avg_free_fidelity = avg_free_fidelity
                        .iter()
                        .zip(&obsv)
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>();
                }
            }

            bar.finish();
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                bar.inc(1);
                let x0 = random_pure_state::<na::U8>(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback::new(
                        h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();
                    let (t_out, rho_out, dy_out) = solver.results().get();

                    let obsv = rho_out
                        .iter()
                        .map(|rho| fidelity(rho, &rhod))
                        .collect::<Vec<f64>>();

                    avg_ctrl_fidelity = avg_ctrl_fidelity
                        .iter()
                        .zip(&obsv)
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>();
                }
            }

            bar.finish();
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                bar.inc(1);
                let x0 = random_pure_state::<na::U8>(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();
                    let (t_out, rho_out, dy_out) = solver.results().get();

                    let obsv = rho_out
                        .iter()
                        .map(|rho| fidelity(rho, &rhod))
                        .collect::<Vec<f64>>();

                    avg_time_fidelity1 = avg_time_fidelity1
                        .iter()
                        .zip(&obsv)
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>();
                }
            }

            bar.finish();
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                bar.inc(1);
                let x0 = random_pure_state::<na::U8>(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::idealmultilevelcompletefeedback::Feedback::new(
                        h, l, hc, f0, f1, y1, delta, gamma, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();
                    let (t_out, rho_out, dy_out) = solver.results().get();

                    let obsv = rho_out
                        .iter()
                        .map(|rho| fidelity(rho, &rhod))
                        .collect::<Vec<f64>>();

                    avg_ideal_fidelity = avg_ideal_fidelity
                        .iter()
                        .zip(&obsv)
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>();
                }
            }

            bar.finish();
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                bar.inc(1);
                let x0 = random_pure_state::<na::U8>(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k2, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();
                    let (t_out, rho_out, dy_out) = solver.results().get();

                    let obsv = rho_out
                        .iter()
                        .map(|rho| fidelity(rho, &rhod))
                        .collect::<Vec<f64>>();

                    avg_time_fidelity2 = avg_time_fidelity2
                        .iter()
                        .zip(&obsv)
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>();
                }
            }

            bar.finish();
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                bar.inc(1);
                let x0 = random_pure_state::<na::U8>(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k3, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();
                    let (t_out, rho_out, dy_out) = solver.results().get();

                    let obsv = rho_out
                        .iter()
                        .map(|rho| fidelity(rho, &rhod))
                        .collect::<Vec<f64>>();

                    avg_time_fidelity3 = avg_time_fidelity3
                        .iter()
                        .zip(&obsv)
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>();
                }
            }

            bar.finish();
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                bar.inc(1);
                let x0 = random_pure_state::<na::U8>(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k4, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();
                    let (t_out, rho_out, dy_out) = solver.results().get();

                    let obsv = rho_out
                        .iter()
                        .map(|rho| fidelity(rho, &rhod))
                        .collect::<Vec<f64>>();

                    avg_time_fidelity4 = avg_time_fidelity4
                        .iter()
                        .zip(&obsv)
                        .map(|(x, y)| x + y)
                        .collect::<Vec<f64>>();
                }
            }

            bar.finish();
        });
    });

    let t_out = (0..=num_steps)
        .map(|n| (n as f64) * dt)
        .collect::<Vec<f64>>();
    let avg_free_fidelity = avg_free_fidelity
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_ctrl_fidelity = avg_ctrl_fidelity
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_time_fidelity1 = avg_time_fidelity1
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_ideal_fidelity = avg_ideal_fidelity
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_time_fidelity2 = avg_time_fidelity2
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_time_fidelity3 = avg_time_fidelity3
        .iter()
        .map(|f| f / (num_inner_tries as f64 * num_tries as f64))
        .collect::<Vec<f64>>();

    let avg_time_fidelity4 = avg_time_fidelity4
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

    let mut time_curve1 = plotpy::Curve::new();
    time_curve1
        .set_label(&format!("Windowed evolution, k = {}", k1))
        .draw(&t_out, &avg_time_fidelity1);

    let mut ideal_curve = plotpy::Curve::new();
    ideal_curve
        .set_label("Ideal evolution")
        .draw(&t_out, &avg_ideal_fidelity);

    let mut time_curve2 = plotpy::Curve::new();
    time_curve2
        .set_label(&format!("Windowed evolution, k = {}", k2))
        .draw(&t_out, &avg_time_fidelity2);

    let mut time_curve3 = plotpy::Curve::new();
    time_curve3
        .set_label(&format!("Windowed evolution, k = {}", k3))
        .draw(&t_out, &avg_time_fidelity3);

    let mut time_curve4 = plotpy::Curve::new();
    time_curve4
        .set_label(&format!("Windowed evolution, k = {}", k4))
        .draw(&t_out, &avg_time_fidelity4);

    plot.add(&free_curve)
        .add(&ctrl_curve)
        .add(&time_curve1)
        .add(&ideal_curve)
        .add(&time_curve2)
        .add(&time_curve3)
        .add(&time_curve4)
        .legend();

    constrainedlayout("Images/parallel2", &mut plot, true)
}
