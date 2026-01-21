use crate::plots::constrainedlayout;
use crate::solver::{Rk4, StochasticSolver, StochasticSystem};
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

const NUMTHREADS: usize = 7;

/*
pub fn parallel_3d() {
    let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let f1 = na::Matrix3::zeros();

    let rhod = na::Matrix3::from_diagonal(&na::Vector3::new(1., 0., 0.)).cast();

    let state_gen = random_pure_state::<na::U3>;

    let num_tries = 1000;
    let num_inner_tries = 20;
    let final_time: f64 = 30.0;
    let dt = 0.0001;
    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();

    let mut avg_free_fidelity = vec![0.; num_steps + 1];
    let mut avg_ideal_fidelity = vec![0.; num_steps + 1];
    let mut avg_ctrl_fidelity = vec![0.; num_steps + 1];
    let mut avg_time_fidelity1 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity2 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity3 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity4 = vec![0.; num_steps + 1];

    let delta = 3.;
    let gamma = 0.2 * delta;
    let y1 = -2.;
    let epsilon = delta * 1.;
    let beta = 0.6;
    let k1 = 5000;
    let k2 = 20000;
    let k3 = 50000;
    let k4 = 100000;

    // let bar = ProgressBar::new(7 * num_tries).with_style(
    //     ProgressStyle::default_bar()
    //         .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
    //         .unwrap(),
    // );

    rayon::scope(|s| {
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::sse::SSE::new(h, l, &mut rng);

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_free_fidelity = sum_arrays(&avg_free_fidelity, &obsv);
                }
                // bar.inc(1);
            }
            avg_free_fidelity = time_average(&avg_free_fidelity, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback::new(
                        h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_ctrl_fidelity = sum_arrays(&avg_ctrl_fidelity, &obsv);
                }
                // bar.inc(1);
            }
            avg_ctrl_fidelity = time_average(&avg_ctrl_fidelity, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::idealmultilevelcompletefeedback::Feedback::new(
                        h, l, hc, f0, f1, y1, delta, gamma, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_ideal_fidelity = sum_arrays(&avg_ideal_fidelity, &obsv);
                }
                // bar.inc(1);
            }
            avg_ideal_fidelity = time_average(&avg_ideal_fidelity, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity1 = sum_arrays(&avg_time_fidelity1, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity1 = time_average(&avg_time_fidelity1, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k2, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity2 = sum_arrays(&avg_time_fidelity2, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity2 = time_average(&avg_time_fidelity2, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k3, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity3 = sum_arrays(&avg_time_fidelity3, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity3 = time_average(&avg_time_fidelity3, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k4, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity4 = sum_arrays(&avg_time_fidelity4, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity4 = time_average(&avg_time_fidelity4, num_tries * num_inner_tries);
        });
    });
    // bar.finish();

    let t_out = (0..=num_steps)
        .map(|n| (n as f64) * dt)
        .collect::<Vec<f64>>();

    println!("Saving to file");
    let mut file = File::create("./3d.csv").expect("Could not create file");

    let mut df: DataFrame = df!(
        "time" => t_out,
        "avg_free_fidelity" => avg_free_fidelity,
        "avg_ideal_fidelity" => avg_ideal_fidelity,
        "avg_ctrl_fidelity" => avg_ctrl_fidelity,
        "avg_time_fidelity1" => avg_time_fidelity1,
        "avg_time_fidelity2" => avg_time_fidelity2,
        "avg_time_fidelity3" => avg_time_fidelity3,
        "avg_time_fidelity4" => avg_time_fidelity4,
    )
    .unwrap();

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df);
}

pub fn parallel_heis() {
    let h = ferromagnetictriangle(&vec![0.5, 0.5, 3.0]);
    let l = h.clone();
    let hc = crate::utils::Operator::<na::U8>::zeros();
    let f1 = crate::utils::Operator::<na::U8>::zeros();

    let eigen = h.symmetric_eigen();
    let f0 = eigen.eigenvectors
        * crate::utils::Operator::<na::U8>::from_fn(|i, j| {
            if i == j - 1 || i == j + 1 {
                na::Complex::ONE
            } else {
                na::Complex::ZERO
            }
        })
        .scale(4.)
        * eigen.eigenvectors.adjoint();
    let rhod =
        crate::utils::Operator::from_diagonal(&na::vector![1., 0., 0., 0., 0., 0., 0., 1.].cast());

    let state_gen = random_pure_state::<na::U8>;

    let num_tries = 1000;
    let num_inner_tries = 20;
    let final_time: f64 = 30.0;
    let dt = 0.0001;
    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();

    let mut avg_free_fidelity = vec![0.; num_steps + 1];
    let mut avg_ideal_fidelity = vec![0.; num_steps + 1];
    let mut avg_ctrl_fidelity = vec![0.; num_steps + 1];
    let mut avg_time_fidelity1 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity2 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity3 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity4 = vec![0.; num_steps + 1];

    let delta = 10.;
    let gamma = 0.2 * delta;
    let y1 = 2. * eigen.eigenvalues.min();
    let epsilon = delta * 1.;
    let beta = 0.6;
    let k1 = 5000;
    let k2 = 20000;
    let k3 = 50000;
    let k4 = 100000;

    // let bar = ProgressBar::new(7 * num_tries).with_style(
    //     ProgressStyle::default_bar()
    //         .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
    //         .unwrap(),
    // );

    rayon::scope(|s| {
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::sse::SSE::new(h, l, &mut rng);

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_free_fidelity = sum_arrays(&avg_free_fidelity, &obsv);
                }
                // bar.inc(1);
            }
            avg_free_fidelity = time_average(&avg_free_fidelity, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback::new(
                        h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_ctrl_fidelity = sum_arrays(&avg_ctrl_fidelity, &obsv);
                }
                // bar.inc(1);
            }
            avg_ctrl_fidelity = time_average(&avg_ctrl_fidelity, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::idealmultilevelcompletefeedback::Feedback::new(
                        h, l, hc, f0, f1, y1, delta, gamma, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_ideal_fidelity = sum_arrays(&avg_ideal_fidelity, &obsv);
                }
                // bar.inc(1);
            }
            avg_ideal_fidelity = time_average(&avg_ideal_fidelity, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity1 = sum_arrays(&avg_time_fidelity1, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity1 = time_average(&avg_time_fidelity1, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k2, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity2 = sum_arrays(&avg_time_fidelity2, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity2 = time_average(&avg_time_fidelity2, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k3, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity3 = sum_arrays(&avg_time_fidelity3, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity3 = time_average(&avg_time_fidelity3, num_tries * num_inner_tries);
        });
        s.spawn(|s| {
            for i in 0..num_tries {
                let x0 = state_gen(Some(i));

                for j in 0..num_inner_tries {
                    let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
                    let mut system = systems::multilevelcompletefeedback::Feedback2::new(
                        h, l, hc, f0, f1, y1, k4, delta, gamma, beta, epsilon, &mut rng,
                    );

                    let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
                    solver.integrate();

                    let rho_out = solver.state_out();
                    let obsv = compute_fidelity(rho_out, &rhod);

                    avg_time_fidelity4 = sum_arrays(&avg_time_fidelity4, &obsv);
                }
                // bar.inc(1);
            }
            avg_time_fidelity4 = time_average(&avg_time_fidelity4, num_tries * num_inner_tries);
        });
    });
    // bar.finish();

    let t_out = (0..=num_steps)
        .map(|n| (n as f64) * dt)
        .collect::<Vec<f64>>();

    println!("Saving to file");
    let mut file = File::create("./heis.csv").expect("Could not create file");

    let mut df: DataFrame = df!(
        "time" => t_out,
        "avg_free_fidelity" => avg_free_fidelity,
        "avg_ideal_fidelity" => avg_ideal_fidelity,
        "avg_ctrl_fidelity" => avg_ctrl_fidelity,
        "avg_time_fidelity1" => avg_time_fidelity1,
        "avg_time_fidelity2" => avg_time_fidelity2,
        "avg_time_fidelity3" => avg_time_fidelity3,
        "avg_time_fidelity4" => avg_time_fidelity4,
    )
    .unwrap();

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df);
}
*/

pub fn parallel_anti_heis() {
    let h = -ferromagnetictriangle(&vec![1.0, 1.0, 2.0]);
    let l = h.clone();
    let hc = crate::utils::Operator::<na::U8>::zeros();
    let f1 = crate::utils::Operator::<na::U8>::zeros();

    let eigen = h.symmetric_eigen();
    let f0 = eigen.eigenvectors
        * crate::utils::Operator::<na::U8>::from_fn(|i, j| {
            if i == j - 1 || i == j + 1 {
                na::Complex::ONE
            } else {
                na::Complex::ZERO
            }
        })
        .scale(4.)
        * eigen.eigenvectors.adjoint();

    let rhod = eigen.eigenvectors
        * crate::utils::Operator::<na::U8>::from_fn(|i, j| {
            if i == j && eigen.eigenvalues[i] < 0. {
                na::Complex::ONE
            } else {
                na::Complex::ZERO
            }
        })
        * eigen.eigenvectors.adjoint();

    let state_gen = random_pure_state::<na::U8>;

    let num_tries = 1000;
    let num_inner_tries = 20;
    // let final_time: f64 = 30.0;
    let final_time: f64 = 15.0;
    let dt = 0.0001;
    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();

    let mut avg_free_fidelity = vec![0.; num_steps + 1];
    let mut avg_ideal_fidelity = vec![0.; num_steps + 1];
    let mut avg_ctrl_fidelity = vec![0.; num_steps + 1];
    let mut avg_time_fidelity1 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity2 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity3 = vec![0.; num_steps + 1];
    let mut avg_time_fidelity4 = vec![0.; num_steps + 1];

    let delta = 6.;
    let gamma = 0.2 * delta;
    let y1 = 2. * eigen.eigenvalues.min();
    let epsilon = delta * 1.;
    let beta = 0.6;
    let k1 = 5000;
    let k2 = 20000;
    let k3 = 50000;
    let k4 = 100000;

    let mut rng1 = StdRng::seed_from_u64(0);
    let mut rng2 = StdRng::seed_from_u64(0);
    let mut rng3 = StdRng::seed_from_u64(0);
    let mut rng4 = StdRng::seed_from_u64(0);
    let mut rng5 = StdRng::seed_from_u64(0);
    let mut rng6 = StdRng::seed_from_u64(0);
    let mut rng7 = StdRng::seed_from_u64(0);

    let sys1 = systems::sse::SSE::new(h, l);
    let sys2 = systems::multilevelcompletefeedback::Feedback::new(
        h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon,
    );
    let sys3 =
        systems::idealmultilevelcompletefeedback::Feedback::new(h, l, hc, f0, f1, y1, delta, gamma);
    let sys4 = systems::multilevelcompletefeedback::Feedback2::new(
        h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon,
    );
    let sys5 = systems::multilevelcompletefeedback::Feedback2::new(
        h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon,
    );
    let sys6 = systems::multilevelcompletefeedback::Feedback2::new(
        h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon,
    );
    let sys7 = systems::multilevelcompletefeedback::Feedback2::new(
        h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon,
    );

    let mut pairs: Vec<Option<(Arc<Mutex<dyn StochasticSystem<State<na::U8>>>>, Vec<f64>)>> = vec![
        Some((Arc::new(Mutex::new(sys1)), avg_free_fidelity.clone())),
        Some((Arc::new(Mutex::new(sys2)), avg_ctrl_fidelity.clone())),
        Some((Arc::new(Mutex::new(sys3)), avg_ideal_fidelity.clone())),
        Some((Arc::new(Mutex::new(sys4)), avg_time_fidelity1.clone())),
        Some((Arc::new(Mutex::new(sys5)), avg_time_fidelity2.clone())),
        Some((Arc::new(Mutex::new(sys6)), avg_time_fidelity3.clone())),
        Some((Arc::new(Mutex::new(sys7)), avg_time_fidelity4.clone())),
    ];

    let mut references = vec![
        Some(&mut avg_free_fidelity),
        Some(&mut avg_ideal_fidelity),
        Some(&mut avg_ctrl_fidelity),
        Some(&mut avg_time_fidelity1),
        Some(&mut avg_time_fidelity2),
        Some(&mut avg_time_fidelity3),
        Some(&mut avg_time_fidelity4),
    ];

    rayon::scope(|s| {
        for l in 0..7 {
            let (mut system, mut avgv) = pairs[l].take().unwrap();
            let mut reference = references[l].take().unwrap();
            s.spawn(|s| {
                let system = system
                    .get_mut()
                    .expect(&format!("Could not take system {}", l));

                for i in 0..num_tries {
                    let x0 = state_gen(Some(i));

                    for j in 0..num_inner_tries {
                        let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);

                        let mut solver = StochasticSolver::new(system, 0.0, x0, final_time, dt);
                        solver.integrate(&mut rng);

                        let rho_out = solver.state_out();
                        let obsv = compute_fidelity(rho_out, &rhod);

                        avgv = sum_arrays(&avgv, &obsv);
                    }
                }
                *reference = time_average(&avgv, num_tries * num_inner_tries);
            })
        }
    });

    // rayon::scope(|s| {
    //     s.spawn(|s| {
    //         for i in 0..num_tries {
    //             let x0 = state_gen(Some(i));
    //
    //             for j in 0..num_inner_tries {
    //                 let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
    //                 let mut system = systems::sse::SSE::new(h, l, &mut rng);
    //
    //                 let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    //                 solver.integrate();
    //
    //                 let rho_out = solver.state_out();
    //                 let obsv = compute_fidelity(rho_out, &rhod);
    //
    //                 avg_free_fidelity = sum_arrays(&avg_free_fidelity, &obsv);
    //             }
    //         }
    //         avg_free_fidelity = time_average(&avg_free_fidelity, num_tries * num_inner_tries);
    //     });
    //     s.spawn(|s| {
    //         for i in 0..num_tries {
    //             let x0 = state_gen(Some(i));
    //
    //             for j in 0..num_inner_tries {
    //                 let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
    //                 let mut system = systems::multilevelcompletefeedback::Feedback::new(
    //                     h, l, hc, f0, f1, y1, delta, gamma, beta, epsilon, &mut rng,
    //                 );
    //
    //                 let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    //                 solver.integrate();
    //
    //                 let rho_out = solver.state_out();
    //                 let obsv = compute_fidelity(rho_out, &rhod);
    //
    //                 avg_ctrl_fidelity = sum_arrays(&avg_ctrl_fidelity, &obsv);
    //             }
    //         }
    //         avg_ctrl_fidelity = time_average(&avg_ctrl_fidelity, num_tries * num_inner_tries);
    //     });
    //     s.spawn(|s| {
    //         for i in 0..num_tries {
    //             let x0 = state_gen(Some(i));
    //
    //             for j in 0..num_inner_tries {
    //                 let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
    //                 let mut system = systems::idealmultilevelcompletefeedback::Feedback::new(
    //                     h, l, hc, f0, f1, y1, delta, gamma, &mut rng,
    //                 );
    //
    //                 let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    //                 solver.integrate();
    //
    //                 let rho_out = solver.state_out();
    //                 let obsv = compute_fidelity(rho_out, &rhod);
    //
    //                 avg_ideal_fidelity = sum_arrays(&avg_ideal_fidelity, &obsv);
    //             }
    //         }
    //         avg_ideal_fidelity = time_average(&avg_ideal_fidelity, num_tries * num_inner_tries);
    //     });
    //     s.spawn(|s| {
    //         for i in 0..num_tries {
    //             let x0 = state_gen(Some(i));
    //
    //             for j in 0..num_inner_tries {
    //                 let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
    //                 let mut system = systems::multilevelcompletefeedback::Feedback2::new(
    //                     h, l, hc, f0, f1, y1, k1, delta, gamma, beta, epsilon, &mut rng,
    //                 );
    //
    //                 let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    //                 solver.integrate();
    //
    //                 let rho_out = solver.state_out();
    //                 let obsv = compute_fidelity(rho_out, &rhod);
    //
    //                 avg_time_fidelity1 = sum_arrays(&avg_time_fidelity1, &obsv);
    //             }
    //         }
    //         avg_time_fidelity1 = time_average(&avg_time_fidelity1, num_tries * num_inner_tries);
    //     });
    //     s.spawn(|s| {
    //         for i in 0..num_tries {
    //             let x0 = state_gen(Some(i));
    //
    //             for j in 0..num_inner_tries {
    //                 let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
    //                 let mut system = systems::multilevelcompletefeedback::Feedback2::new(
    //                     h, l, hc, f0, f1, y1, k2, delta, gamma, beta, epsilon, &mut rng,
    //                 );
    //
    //                 let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    //                 solver.integrate();
    //
    //                 let rho_out = solver.state_out();
    //                 let obsv = compute_fidelity(rho_out, &rhod);
    //
    //                 avg_time_fidelity2 = sum_arrays(&avg_time_fidelity2, &obsv);
    //             }
    //         }
    //         avg_time_fidelity2 = time_average(&avg_time_fidelity2, num_tries * num_inner_tries);
    //     });
    //     s.spawn(|s| {
    //         for i in 0..num_tries {
    //             let x0 = state_gen(Some(i));
    //
    //             for j in 0..num_inner_tries {
    //                 let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
    //                 let mut system = systems::multilevelcompletefeedback::Feedback2::new(
    //                     h, l, hc, f0, f1, y1, k3, delta, gamma, beta, epsilon, &mut rng,
    //                 );
    //
    //                 let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    //                 solver.integrate();
    //
    //                 let rho_out = solver.state_out();
    //                 let obsv = compute_fidelity(rho_out, &rhod);
    //
    //                 avg_time_fidelity3 = sum_arrays(&avg_time_fidelity3, &obsv);
    //             }
    //         }
    //         avg_time_fidelity3 = time_average(&avg_time_fidelity3, num_tries * num_inner_tries);
    //     });
    //     s.spawn(|s| {
    //         for i in 0..num_tries {
    //             let x0 = state_gen(Some(i));
    //
    //             for j in 0..num_inner_tries {
    //                 let mut rng = StdRng::seed_from_u64(num_inner_tries * i + j);
    //                 let mut system = systems::multilevelcompletefeedback::Feedback2::new(
    //                     h, l, hc, f0, f1, y1, k4, delta, gamma, beta, epsilon, &mut rng,
    //                 );
    //
    //                 let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    //                 solver.integrate();
    //
    //                 let rho_out = solver.state_out();
    //                 let obsv = compute_fidelity(rho_out, &rhod);
    //
    //                 avg_time_fidelity4 = sum_arrays(&avg_time_fidelity4, &obsv);
    //             }
    //         }
    //         avg_time_fidelity4 = time_average(&avg_time_fidelity4, num_tries * num_inner_tries);
    //     });
    // });

    let t_out = (0..=num_steps)
        .map(|n| (n as f64) * dt)
        .collect::<Vec<f64>>();

    println!("Saving to file");
    let mut file = File::create("./anti_heis.csv").expect("Could not create file");

    let mut df: DataFrame = df!(
        "time" => t_out,
        "avg_free_fidelity" => avg_free_fidelity,
        "avg_ideal_fidelity" => avg_ideal_fidelity,
        "avg_ctrl_fidelity" => avg_ctrl_fidelity,
        "avg_time_fidelity1" => avg_time_fidelity1,
        "avg_time_fidelity2" => avg_time_fidelity2,
        "avg_time_fidelity3" => avg_time_fidelity3,
        "avg_time_fidelity4" => avg_time_fidelity4,
    )
    .unwrap();

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df);
}
