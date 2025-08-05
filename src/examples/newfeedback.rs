use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn newfeedback() -> SolverResult<()> {
    let mut plot1 = plotpy::Plot::new();
    // let mut plot2 = plotpy::Plot::new();
    // let mut plot3 = plotpy::Plot::new();
    // let mut plot4 = plotpy::Plot::new();

    // plot2.set_gridspec("a", 2, 2, "width_ratios=[1,1], height_ratios=[1,1]");
    plots::plot_bloch_sphere(plot1.set_subplot_3d(2, 2, 1))?;
    plots::plot_bloch_sphere(plot1.set_subplot_3d(2, 2, 2))?;
    plots::plot_bloch_sphere(plot1.set_subplot_3d(2, 2, 3))?;
    plots::plot_bloch_sphere(plot1.set_subplot_3d(2, 2, 4))?;

    // let h = PAULI_Z + PAULI_X;
    // let l = PAULI_X;
    // let f = QubitOperator::new(
    //     na::Complex::ZERO,
    //     -na::Complex::I,
    //     na::Complex::I,
    //     na::Complex::new(2., 0.),
    // );

    let h = PAULI_Z + PAULI_Y.scale(0.10);
    let l = PAULI_Z + PAULI_X.scale(0.10);
    let f1 = PAULI_Y.scale(0.10);
    // let h = PAULI_Z;
    // let l = PAULI_X;
    // let f0s = vec![
    //     na::Matrix2::new(0.0, 0.0, 0.0, 0.0).cast(),
    //     na::Matrix2::new(0.0, 1.0, 1.0, 0.0).cast(),
    //     na::Matrix2::new(100.0, 0.0, 0.0, 1.0).cast(),
    //     na::Matrix2::new(1.0, 0.0, 0.0, 100.0).cast(),
    // ];
    // let f1 = PAULI_Y;

    // let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    // let x0 = random_qubit_state();
    let x0 = random_pure_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 4;
    let final_time: f64 = 3.0;
    let dt = 0.0001;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];
    //
    // let mut mean_traj = vec![BlochVector::zeros(); (final_time / dt).ceil().to_usize().unwrap()];
    //
    // let bar = ProgressBar::new(num_tries).with_style(
    //     ProgressStyle::default_bar()
    //         .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
    //         .unwrap(),
    // );
    let rho0 = na::Matrix2::new(1.0, 0.0, 0.0, 0.0).cast();
    let rho1 = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    let y0 = ((l + l.adjoint()) * rho0).trace().re;
    let y1 = ((l + l.adjoint()) * rho1).trace().re;
    let lb = 0.1; // no sense in this check
    let ub = 3.;
    let lbe = (y0 - y1).abs() * lb;
    let ube = (y0 - y1).abs() * ub;
    let alpha = 0.99;

    for i in 0..num_tries {
        // bar.inc(1);
        // let mut rng = StdRng::seed_from_u64(0);
        let mut rng = rand::rng();
        let mut system = if i < 2 {
            systems::qubitnewfeedbackv1::QubitNewFeedbackV1::new(
                h, l, PAULI_X, f1, y0, y1, lb, ub, alpha, &mut rng,
            )
        } else {
            systems::qubitnewfeedbackv1::QubitNewFeedbackV1::new(
                h,
                l,
                QubitOperator::zeros(),
                f1,
                y0,
                y1,
                lb,
                ub,
                alpha,
                &mut rng,
            )
        };
        // println!("{0}", system.tf);

        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();
        // let y_out: Vec<f64> = dy_out
        //     .into_iter()
        //     .scan(0., |acc, x| {
        //         *acc += x;
        //         Some(*acc)
        //     })
        //     .collect();
        //
        // let mut corr = vec![0.; y_out.len()];
        // let mut acc = 0.;
        // let mut t = 0.;
        // for i in 0..dy_out.len() - 1 {
        //     t += dt;
        //     corr[i + 1] = if (y_out[i + 1] / t - y0).abs() > (y_out[i + 1] / t - y1).abs() {
        //         0.
        //     } else {
        //         -1.
        //     };
        // }

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

        plot1.set_subplot_3d(2, 2, i + 1).add(&trajectory);

        let mut start = plotpy::Curve::new();
        start
            .set_line_color("#000000")
            .set_marker_style("o")
            .set_marker_size(10.0)
            .points_3d_begin()
            .points_3d_add(x0bloch[0], x0bloch[1], x0bloch[2])
            .points_3d_end();

        plot1.set_subplot_3d(2, 2, i + 1).add(&start);

        let last = obsv.last().unwrap();
        let mut end = plotpy::Curve::new();

        end.set_line_color("#FFFF00") //(colors[i as usize])
            .set_marker_style("o")
            .set_marker_size(10.0)
            .points_3d_begin()
            .points_3d_add(last[0], last[1], last[2])
            .points_3d_end();

        plot1.set_subplot_3d(2, 2, i + 1).add(&end);
        plot1.set_subplot_3d(2, 2, i + 1).set_equal_axes(true);

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
        //
        // plot2
        //     .set_subplot(2, 2, i + 1)
        //     .add(&xaxis)
        //     .add(&yaxis)
        //     .add(&zaxis);
        //
        // let mut y = plotpy::Curve::new();
        // y.set_line_color(colors[i as usize]).draw(t_out, &y_out);
        // plot3.set_subplot(2, 2, i + 1).add(&y);
        //
        // let mut corrs = plotpy::Curve::new();
        // corrs.set_line_color(colors[i as usize]).draw(t_out, &corr);
        // plot4.set_subplot(2, 2, i + 1).add(&corrs);

        //     mean_traj = mean_traj
        //         .iter()
        //         .zip(obsv)
        //         .map(|(x, y)| x + y.scale(1. / num_tries as f64))
        //         .collect::<Vec<BlochVector>>();
    }
    // bar.finish();

    // let mut mean_curve = plotpy::Curve::new();
    // mean_curve.set_line_color("#ff0000").draw_3d(
    //     &mean_traj.iter().map(|o| o[0]).collect::<Vec<f64>>(),
    //     &mean_traj.iter().map(|o| o[1]).collect::<Vec<f64>>(),
    //     &mean_traj.iter().map(|o| o[2]).collect::<Vec<f64>>(),
    // );
    //
    // plot.add(&mean_curve);
    //
    // let mut endpoint = plotpy::Curve::new();
    // endpoint
    //     .set_line_color("#000000")
    //     .set_marker_style("o")
    //     .set_marker_size(10.0)
    //     .points_3d_begin()
    //     .points_3d_add(
    //         mean_traj.last().unwrap()[0],
    //         mean_traj.last().unwrap()[1],
    //         mean_traj.last().unwrap()[2],
    //     )
    //     .points_3d_end();
    //
    // plot.add(&endpoint);

    plot1.show("tempimages")?;
    // plot2.show("tempimages")?;
    // plot3.show("tempimages")?;
    // plot4.show("tempimages")?;

    Ok(())
}
