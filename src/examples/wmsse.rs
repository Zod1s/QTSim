use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn wmsse() -> SolverResult<()> {
    // let h = PAULI_Z;
    // let l = PAULI_X;
    // let f = PAULI_Y;
    let h = PAULI_Z + PAULI_Y.scale(0.30);
    let l = PAULI_Z + PAULI_X.scale(0.30);
    let f = PAULI_Y.scale(0.30);
    // let h = PAULI_Z + PAULI_Y;
    // let l = PAULI_X + PAULI_Z;
    // let h = PAULI_Z + PAULI_X;
    // let l = PAULI_X;
    // let f = QubitOperator::new(
    //     na::Complex::ZERO,
    //     -na::Complex::I,
    //     na::Complex::I,
    //     na::Complex::new(2., 0.),
    // );

    // let mut rng = StdRng::seed_from_u64(0);
    let mut rng = rand::rng();
    let mut system = systems::qubitwisemansse::QubitWisemanSSE::new(h, l, f, &mut rng);
    // let x0 = random_qubit_state();
    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast::<na::Complex<f64>>();
    // let x0 = random_pure_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time = 30.;
    let dt = 0.001;

    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot)?;

    let mut plot2 = plotpy::Plot::new();

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    // let mut mean_traj = vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for i in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, _) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(to_bloch_unchecked)
            .collect::<Vec<BlochVector>>();

        let mut trajectory = plotpy::Curve::new();
        trajectory.set_line_color(colors[i as usize]).draw_3d(
            &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
            &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
        );

        plot.add(&trajectory);

        let last = obsv.last().unwrap();
        let mut end = plotpy::Curve::new();

        end.set_line_color(colors[i as usize])
            .set_marker_style("o")
            .set_marker_size(10.0)
            .points_3d_begin()
            .points_3d_add(last[0], last[1], last[2])
            .points_3d_end();

        plot.add(&end);

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
        let mut zaxis = plotpy::Curve::new();
        zaxis
            .set_line_color(colors[i as usize])
            .draw(t_out, &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>());

        plot2.add(&zaxis);

        // mean_traj = mean_traj
        //     .iter()
        //     .zip(obsv)
        //     .map(|(x, y)| x + y.scale(1. / num_tries as f64))
        //     .collect::<Vec<BlochVector>>();
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

    // let mut mean_curve = plotpy::Curve::new();
    // mean_curve.set_line_color("#ff0000").draw_3d(
    //     &mean_traj.iter().map(|o| o[0]).collect::<Vec<f64>>(),
    //     &mean_traj.iter().map(|o| o[1]).collect::<Vec<f64>>(),
    //     &mean_traj.iter().map(|o| o[2]).collect::<Vec<f64>>(),
    // );
    //
    // plot.add(&mean_curve);

    // let system = systems::qubitwisemanfme::QubitWisemanFME::new(h, l, f);
    // let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
    // solver.integrate()?;
    //
    // let (_, rho_out) = solver.results().get();
    //
    // let obsv = rho_out
    //     .iter()
    //     .map(to_bloch_unchecked)
    //     .collect::<Vec<BlochVector>>();
    //
    // let mut trajectory = plotpy::Curve::new();
    // trajectory
    //     .set_line_color("#000000")
    //     .set_line_width(2.0)
    //     .draw_3d(
    //         &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
    //         &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
    //         &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
    //     );
    //
    // plot.add(&trajectory);

    plot.set_equal_axes(true).show("tempimages")?;
    plot2.show("tempimages")?;

    Ok(())
}
