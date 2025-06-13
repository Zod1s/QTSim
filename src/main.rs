#![allow(unused)]

mod plots;
mod solver;
mod utils;
mod wiener;

use std::f64::consts::PI;

use crate::solver::{Rk4, StochasticSolver, StochasticSystem, System};
use crate::utils::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use num_cpus;
use plotpy;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;
use rayon::prelude::*;
use std::process::Command;

fn main() -> utils::SolverResult<()> {
    comparison()?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct QubitWisemanFME {
    h: QubitOperator,
    l: QubitOperator,
    f: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
}

impl QubitWisemanFME {
    pub fn new(h: QubitOperator, l: QubitOperator, f: QubitOperator) -> Self {
        let lhat = l - f * na::Complex::I;
        let hhat = h + (f * l + l.adjoint() * f).scale(0.5);
        Self {
            h,
            l,
            f,
            hhat,
            lhat,
        }
    }
}

impl System<QubitState> for QubitWisemanFME {
    fn system(&self, t: f64, rho: &QubitState, drho: &mut QubitState) {
        *drho = -commutator(&self.hhat, rho) * na::Complex::I
            + self.lhat * rho * self.lhat.adjoint()
            - anticommutator(&(self.lhat.adjoint() * self.lhat), rho).scale(0.5);
    }
}

fn wmfme() -> utils::SolverResult<()> {
    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    let system = QubitWisemanFME::new(h, l, f);

    let mut sphere = plotpy::Surface::new();
    sphere
        .set_surf_color("#00000020")
        .draw_sphere(&[0.0, 0.0, 0.0], 1.0, 40, 40)?;

    let mut plot = plotpy::Plot::new();
    plot.add(&sphere).set_equal_axes(true);

    let colors = vec![
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff00ff", "#e1ff00",
    ];

    for i in 0..10 {
        let x0 = random_qubit_state();

        let mut solver = Rk4::new(system, 0.0, x0, 4.0, 0.001);
        solver.integrate()?;

        let (t_out, rho_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| to_bloch_unchecked(rho))
            .collect::<Vec<BlochVector>>();

        let mut trajectory = plotpy::Curve::new();
        trajectory.set_line_color(colors[i]).draw_3d(
            &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        );

        plot.add(&trajectory);

        let mut start = plotpy::Curve::new();
        start
            .set_line_color(colors[i])
            .set_marker_style("o")
            .set_marker_size(10.0)
            .points_3d_begin()
            .points_3d_add(obsv[0][0], obsv[0][1], obsv[0][2])
            .points_3d_end();

        plot.add(&start);
    }

    plot.show("tempimages")?;

    Ok(())
}

#[derive(Debug)]
struct QubitWisemanSSE<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitWisemanSSE<'a, R> {
    pub fn new(h: QubitOperator, l: QubitOperator, f: QubitOperator, rng: &'a mut R) -> Self {
        let lhat = l - f * na::Complex::I;
        let hhat = h + (f * l + l.adjoint() * f).scale(0.5);

        Self {
            h,
            l,
            f,
            hhat,
            lhat,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitWisemanSSE<'a, R> {
    fn system(&self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let id = QubitOperator::identity();
        let fst =
            (self.hhat * na::Complex::I + self.lhat.adjoint() * self.lhat.scale(0.5)).scale(dt);
        let snd = self
            .lhat
            .scale((self.lhat * x + x * self.lhat.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.lhat * self.lhat).scale(dw[0].powi(2) - dt);

        let m = id - fst + snd + thd;

        let num = m * x * m.adjoint();
        *dx = num.scale(1. / num.trace().re) - x;
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}

fn wmsse() -> utils::SolverResult<()> {
    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    let mut rng = StdRng::seed_from_u64(0);
    let mut system = QubitWisemanSSE::new(h, l, f, &mut rng);
    let x0 = random_qubit_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 2.0;
    let dt = 0.001;

    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot)?;

    // let mut plot2 = plotpy::Plot::new();

    let colors = vec![
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let mut mean_traj = vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for i in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| to_bloch_unchecked(rho))
            .collect::<Vec<BlochVector>>();
        // assert!(
        //     rho_out
        //         .iter()
        //         .all(|rho| rho.symmetric_eigenvalues().iter().all(|&e| e >= 0.)),
        //     "eigenvalues are {:?}",
        //     rho_out
        //         .iter()
        //         .map(|rho| rho.symmetric_eigenvalues())
        //         .collect::<Vec<na::Vector2<f64>>>()
        // );
        //
        // let mut trajectory = plotpy::Curve::new();
        // trajectory.set_line_color(colors[i as usize]).draw_3d(
        //     &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
        //     &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
        //     &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        // );
        //
        // plot.add(&trajectory);
        //
        // let mut y_out = vec![0.; dy_out.len()];
        // let mut acc = 0.;
        // for i in 0..dy_out.len() {
        //     acc += dy_out[i];
        //     y_out[i] = acc;
        // }
        //
        // let mut y = plotpy::Curve::new();
        // y.set_line_color(colors[i as usize]).draw(t_out, &y_out);
        // plot2.add(&y);

        mean_traj = mean_traj
            .iter()
            .zip(obsv)
            .map(|(x, y)| x + y.scale(1. / num_tries as f64))
            .collect::<Vec<BlochVector>>();
    }
    bar.finish();

    let mut start = plotpy::Curve::new();
    start
        .set_line_color("#000000")
        .set_marker_style("o")
        .set_marker_size(10.0)
        .points_3d_begin()
        .points_3d_add(x0bloch[0], x0bloch[1], x0bloch[2])
        .points_3d_end();

    plot.add(&start);

    let mut mean_curve = plotpy::Curve::new();
    mean_curve.set_line_color("#ff0000").draw_3d(
        &mean_traj.iter().map(|o| o[0]).collect::<Vec<f64>>(),
        &mean_traj.iter().map(|o| o[1]).collect::<Vec<f64>>(),
        &mean_traj.iter().map(|o| o[2]).collect::<Vec<f64>>(),
    );

    plot.add(&mean_curve);

    let system = QubitWisemanFME::new(h, l, f);
    let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
    solver.integrate()?;

    let (t_out, rho_out) = solver.results().get();

    let obsv = rho_out
        .iter()
        .map(|rho| to_bloch_unchecked(rho))
        .collect::<Vec<BlochVector>>();

    let mut trajectory = plotpy::Curve::new();
    trajectory
        .set_line_color("#000000")
        .set_line_width(2.0)
        .draw_3d(
            &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        );

    plot.add(&trajectory);

    plot.set_equal_axes(true).show("tempimages")?;
    // plot2.set_equal_axes(true).show("tempimages")?;

    Ok(())
}

#[derive(Debug)]
struct QubitSequentialControl<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f: QubitOperator,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitSequentialControl<'a, R> {
    pub fn new(h: QubitOperator, l: QubitOperator, f: QubitOperator, rng: &'a mut R) -> Self {
        Self {
            h,
            l,
            f,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitSequentialControl<'a, R> {
    fn system(&self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let id = QubitOperator::identity();
        let fst = (self.h * na::Complex::I + self.l.adjoint() * self.l.scale(0.5)).scale(dt);
        let snd = self
            .l
            .scale((self.l * x + x * self.l.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.l * self.l).scale(dw[0].powi(2) - dt);

        let m = id - fst + snd + thd;

        let num = m * x * m.adjoint();
        let rho = num.scale(1. / num.trace().re);
        let dy = self.measurement(x, dt, dw[0]);

        let feedback = |sigma| -commutator(&self.f, sigma) * na::Complex::I;

        // I don't know if I am doing this right, it may lose the positivity constraint
        // I am evolving the state for dt, then applying the series expansion of exp(Mdy)
        // to the new state, but I don't know if it makes sense to stop to the second order
        // from a numerical point of view
        // Also, I need to explore whether I need to use dt or dy^2 for the actual dy^2

        let feedrho = feedback(&rho);

        *dx = rho - x + (feedrho).scale(dy) + (feedback(&feedrho)).scale(0.5 * dy.powi(2));
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}

fn wmseq() -> utils::SolverResult<()> {
    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    // let mut rng = StdRng::seed_from_u64(0);
    let mut rng = rand::rng();
    let mut system = QubitSequentialControl::new(h, l, f, &mut rng);
    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast::<na::Complex<f64>>();
    // let x0 = random_qubit_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 2.0;
    let dt = 0.001;

    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot)?;

    // let mut plot2 = plotpy::Plot::new();

    let colors = vec![
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let mut mean_traj = vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for i in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| to_bloch_unchecked(rho))
            .collect::<Vec<BlochVector>>();

        // assert!(
        //     rho_out
        //         .iter()
        //         .all(|rho| rho.symmetric_eigenvalues().iter().all(|&e| e >= 0.)),
        //     "eigenvalues are {:?}",
        //     rho_out
        //         .iter()
        //         .map(|rho| rho.symmetric_eigenvalues())
        //         .collect::<Vec<na::Vector2<f64>>>()
        // );

        // let mut trajectory = plotpy::Curve::new();
        // trajectory.set_line_color(colors[i as usize]).draw_3d(
        //     &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
        //     &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
        //     &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        // );
        //
        // plot.add(&trajectory);

        // let mut y_out = vec![0.; dy_out.len()];
        // let mut acc = 0.;
        // for i in 0..dy_out.len() {
        //     acc += dy_out[i];
        //     y_out[i] = acc;
        // }
        //
        // let mut y = plotpy::Curve::new();
        // y.set_line_color(colors[i as usize]).draw(t_out, &y_out);
        // plot2.add(&y);

        mean_traj = mean_traj
            .iter()
            .zip(obsv)
            .map(|(x, y)| x + y.scale(1. / num_tries as f64))
            .collect::<Vec<BlochVector>>();
    }
    bar.finish();

    let mut start = plotpy::Curve::new();
    start
        .set_line_color("#000000")
        .set_marker_style("o")
        .set_marker_size(10.0)
        .points_3d_begin()
        .points_3d_add(x0bloch[0], x0bloch[1], x0bloch[2])
        .points_3d_end();

    plot.add(&start);

    let mut mean_curve = plotpy::Curve::new();
    mean_curve.set_line_color("#ff0000").draw_3d(
        &mean_traj.iter().map(|o| o[0]).collect::<Vec<f64>>(),
        &mean_traj.iter().map(|o| o[1]).collect::<Vec<f64>>(),
        &mean_traj.iter().map(|o| o[2]).collect::<Vec<f64>>(),
    );

    plot.add(&mean_curve);

    let system = QubitWisemanFME::new(h, l, f);
    let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
    solver.integrate()?;

    let (t_out, rho_out) = solver.results().get();

    let obsv = rho_out
        .iter()
        .map(|rho| to_bloch_unchecked(rho))
        .collect::<Vec<BlochVector>>();

    let mut trajectory = plotpy::Curve::new();
    trajectory
        .set_line_color("#000000")
        .set_line_width(2.0)
        .draw_3d(
            &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        );

    plot.add(&trajectory);

    plot.set_equal_axes(true).show("tempimages")?;

    Ok(())
}

fn comparison() -> utils::SolverResult<()> {
    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot.set_subplot_3d(2, 2, 1))?;
    plots::plot_bloch_sphere(&mut plot.set_subplot_3d(2, 2, 2))?;
    plots::plot_bloch_sphere(&mut plot.set_subplot_3d(2, 2, 3))?;
    plots::plot_bloch_sphere(&mut plot.set_subplot_3d(2, 2, 4))?;

    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    let mut rng1 = StdRng::seed_from_u64(0);
    let mut rng2 = StdRng::seed_from_u64(0);
    // let mut rng1 = rand::rng();
    // let mut rng2 = rand::rng();
    let mut qwsse = QubitWisemanSSE::new(h, l, f, &mut rng1);
    let mut qwseq = QubitSequentialControl::new(h, l, f, &mut rng2);
    let x0 = QubitState::new(
        na::Complex::ONE,
        -na::Complex::I,
        na::Complex::I,
        na::Complex::ONE,
    )
    .scale(0.5);
    // let x0 = random_qubit_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 100;
    let final_time: f64 = 2.0;
    let dt = 0.01;

    let colors = vec![
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let mut mean_traj_sse =
        vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];
    let mut mean_traj_seq =
        vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for i in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut qwsse, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| to_bloch_unchecked(rho))
            .collect::<Vec<BlochVector>>();

        if i % 10 == 0 {
            let mut trajectory = plotpy::Curve::new();
            trajectory
                .set_line_color(colors[(i / 10) as usize])
                .draw_3d(
                    &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
                    &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
                    &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
                );

            plot.set_subplot_3d(2, 2, 1).add(&trajectory);
        }

        mean_traj_sse = mean_traj_sse
            .iter()
            .zip(obsv)
            .map(|(x, y)| x + y.scale(1. / num_tries as f64))
            .collect::<Vec<BlochVector>>();

        let mut solver = StochasticSolver::new(&mut qwseq, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| to_bloch_unchecked(rho))
            .collect::<Vec<BlochVector>>();

        if i % 10 == 0 {
            let mut trajectory = plotpy::Curve::new();
            trajectory
                .set_line_color(colors[(i / 10) as usize])
                .draw_3d(
                    &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
                    &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
                    &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
                );

            plot.set_subplot_3d(2, 2, 2).add(&trajectory);
        }

        mean_traj_seq = mean_traj_seq
            .iter()
            .zip(obsv)
            .map(|(x, y)| x + y.scale(1. / num_tries as f64))
            .collect::<Vec<BlochVector>>();
    }
    bar.finish();

    let mut start = plotpy::Curve::new();
    start
        .set_line_color("#000000")
        .set_marker_style("o")
        .set_marker_size(10.0)
        .points_3d_begin()
        .points_3d_add(x0bloch[0], x0bloch[1], x0bloch[2])
        .points_3d_end();

    plot.set_subplot_3d(2, 2, 1).add(&start);
    plot.set_subplot_3d(2, 2, 2).add(&start);
    plot.set_subplot_3d(2, 2, 3).add(&start);
    plot.set_subplot_3d(2, 2, 4).add(&start);

    let mut mean_curve = plotpy::Curve::new();
    mean_curve.set_line_color("#ff0000").draw_3d(
        &mean_traj_sse.iter().map(|o| o[0]).collect::<Vec<f64>>(),
        &mean_traj_sse.iter().map(|o| o[1]).collect::<Vec<f64>>(),
        &mean_traj_sse.iter().map(|o| o[2]).collect::<Vec<f64>>(),
    );

    plot.set_subplot_3d(2, 2, 3).add(&mean_curve);

    let mut mean_curve = plotpy::Curve::new();
    mean_curve.set_line_color("#ff0000").draw_3d(
        &mean_traj_seq.iter().map(|o| o[0]).collect::<Vec<f64>>(),
        &mean_traj_seq.iter().map(|o| o[1]).collect::<Vec<f64>>(),
        &mean_traj_seq.iter().map(|o| o[2]).collect::<Vec<f64>>(),
    );

    plot.set_subplot_3d(2, 2, 4).add(&mean_curve);

    let system = QubitWisemanFME::new(h, l, f);
    let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
    solver.integrate()?;

    let (t_out, rho_out) = solver.results().get();

    let obsv = rho_out
        .iter()
        .map(|rho| to_bloch_unchecked(rho))
        .collect::<Vec<BlochVector>>();

    let mut trajectory = plotpy::Curve::new();
    trajectory
        .set_line_color("#000000")
        .set_line_width(2.0)
        .draw_3d(
            &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        );

    plot.set_subplot_3d(2, 2, 1).add(&trajectory);
    plot.set_subplot_3d(2, 2, 2).add(&trajectory);
    plot.set_subplot_3d(2, 2, 3).add(&trajectory);
    plot.set_subplot_3d(2, 2, 4).add(&trajectory);

    plot.set_subplot_3d(2, 2, 1)
        .set_title("Shift Version, unravelings")
        .set_equal_axes(true);
    plot.set_subplot_3d(2, 2, 2)
        .set_title("Sequential Version, unravelings")
        .set_equal_axes(true);
    plot.set_subplot_3d(2, 2, 3)
        .set_title("Shift Version, mean trajectory")
        .set_equal_axes(true);
    plot.set_subplot_3d(2, 2, 4)
        .set_title("Sequential Version, mean trajectory")
        .set_equal_axes(true);

    plot.show("tempimage")?;

    Ok(())
}
