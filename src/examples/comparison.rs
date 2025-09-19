use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn comparison() -> SolverResult<()> {
    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(plot.set_subplot_3d(2, 2, 1))?;
    plots::plot_bloch_sphere(plot.set_subplot_3d(2, 2, 2))?;
    plots::plot_bloch_sphere(plot.set_subplot_3d(2, 2, 3))?;
    plots::plot_bloch_sphere(plot.set_subplot_3d(2, 2, 4))?;

    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    let mut rng1 = StdRng::seed_from_u64(0);
    let mut rng2 = StdRng::seed_from_u64(0);
    // let mut rng1 = rand::rng();
    // let mut rng2 = rand::rng();
    let mut qwsse = systems::qubitwisemansse::QubitWisemanSSE::new(h, l, f, &mut rng1);
    let mut qwseq =
        systems::qubitsequentialcontrol::QubitSequentialControl::new(h, l, f, &mut rng2);
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

    let colors = [
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

        let (_, rho_out, _) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(to_bloch_unchecked)
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

        let (_, rho_out, _) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(to_bloch_unchecked)
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

    let system = systems::qubitwisemanfme::QubitWisemanFME::new(h, l, f);
    let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
    solver.integrate()?;

    let (_, rho_out) = solver.results().get();

    let obsv = rho_out
        .iter()
        .map(to_bloch_unchecked)
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

    plot.show("tempimages")?;

    Ok(())
}
