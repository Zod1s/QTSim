use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn wmfme() -> SolverResult<()> {
    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    let system = systems::QubitWisemanFME::new(h, l, f);

    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot);

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff00ff", "#e1ff00",
    ];

    for i in 0..10 {
        let x0 = random_qubit_state();

        let mut solver = Rk4::new(system, 0.0, x0, 4.0, 0.001);
        solver.integrate()?;

        let (_, rho_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(to_bloch_unchecked)
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

pub fn wmsse() -> SolverResult<()> {
    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    let mut rng = StdRng::seed_from_u64(0);
    let mut system = systems::QubitWisemanSSE::new(h, l, f, &mut rng);
    let x0 = random_qubit_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 2.0;
    let dt = 0.001;

    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot)?;

    // let mut plot2 = plotpy::Plot::new();

    // let colors = [
    //     "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
    //     "#ff0037", "#e1ff00",
    // ];

    let mut mean_traj = vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for _ in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (_, rho_out, _) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(to_bloch_unchecked)
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

    let system = systems::QubitWisemanFME::new(h, l, f);
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

    plot.add(&trajectory);

    plot.set_equal_axes(true).show("tempimages")?;
    // plot2.set_equal_axes(true).show("tempimages")?;

    Ok(())
}

pub fn wmseq() -> SolverResult<()> {
    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    // let mut rng = StdRng::seed_from_u64(0);
    let mut rng = rand::rng();
    let mut system = systems::QubitSequentialControl::new(h, l, f, &mut rng);
    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast::<na::Complex<f64>>();
    // let x0 = random_qubit_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 1;
    let final_time: f64 = 10.0;
    let dt = 0.001;

    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot)?;

    // let mut plot2 = plotpy::Plot::new();

    // let colors =[
    //     "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
    //     "#ff0037", "#e1ff00",
    // ];

    let mut mean_traj = vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for _ in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (_, rho_out, _) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(to_bloch_unchecked)
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

    let system = systems::QubitWisemanFME::new(h, l, f);
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

    plot.add(&trajectory);

    let last = obsv.last().unwrap();
    let mut end = plotpy::Curve::new();

    end.set_line_color("#00FF00")
        .set_marker_style("o")
        .set_marker_size(10.0)
        .points_3d_begin()
        .points_3d_add(last[0], last[1], last[2])
        .points_3d_end();

    plot.add(&end);

    plot.set_equal_axes(true).show("tempimages")?;

    Ok(())
}

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
    let mut qwsse = systems::QubitWisemanSSE::new(h, l, f, &mut rng1);
    let mut qwseq = systems::QubitSequentialControl::new(h, l, f, &mut rng2);
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

    let system = systems::QubitWisemanFME::new(h, l, f);
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

    plot.show("tempimage")?;

    Ok(())
}

pub fn newfeedback() -> SolverResult<()> {
    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot)?;

    let h = PAULI_Z;
    let l = PAULI_X;
    let f = PAULI_Y;
    // let rhod = na::Matrix2::new(1., 0., 0., 0.).cast::<na::Complex<f64>>();
    let mut rng = StdRng::seed_from_u64(0);
    let mut system = systems::QubitNewFeedbackV2::new(h, l, f, &mut rng);

    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast::<na::Complex<f64>>();
    // let x0 = random_qubit_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 1;
    let final_time: f64 = 40.0;
    let dt = 0.001;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    // let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
    // solver.integrate();
    // let (t_out, rho_out, _) = solver.results().get();
    // let obsv = rho_out
    //     .iter()
    //     .map(to_bloch_unchecked)
    //     .collect::<Vec<BlochVector>>();

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for i in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (_, rho_out, _) = solver.results().get();

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

    plot.set_equal_axes(true).show("tempimages")?;
    //
    // let mut plot2 = plotpy::Plot::new();
    //
    // let mut xaxis = plotpy::Curve::new();
    // xaxis
    //     .set_line_color("#FF0000")
    //     .draw(t_out, &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>());
    // let mut yaxis = plotpy::Curve::new();
    // yaxis
    //     .set_line_color("#00FF00")
    //     .draw(t_out, &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>());
    // let mut zaxis = plotpy::Curve::new();
    // zaxis
    //     .set_line_color("#0000FF")
    //     .draw(t_out, &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>());
    // plot2.add(&xaxis).add(&yaxis).add(&zaxis);
    // plot2.set_equal_axes(true).show("tempimages")?;

    Ok(())
}
