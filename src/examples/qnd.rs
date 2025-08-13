use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn qnd() -> SolverResult<()> {
    // [\rho_d, L + L^\dagger] = 0 case
    let mut plot1 = plotpy::Plot::new();
    let mut plot2 = plotpy::Plot::new();
    // let mut plot3 = plotpy::Plot::new();

    plots::plot_bloch_sphere(&mut plot1)?;

    let h = PAULI_Z;
    let l = PAULI_Z;
    let f0 = PAULI_X;

    let rho0 = na::Matrix2::new(1.0, 0.0, 0.0, 0.0).cast();
    let rho1 = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    let y0 = ((l + l.adjoint()) * rho0).trace().re;
    let y1 = ((l + l.adjoint()) * rho1).trace().re;

    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    // let x0 = random_qubit_state();
    // let x0 = random_pure_state();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 5.0;
    let dt = 0.0001;

    let mut avg_sigmaz = vec![0.; (final_time / dt).ceil().to_usize().unwrap() + 1];
    let mut converged_traj = 0;

    let lb = 0.1; // no sense in this check
    let ub = 3.;
    let lbe = (y0 - y1).abs() * lb;
    let ube = (y0 - y1).abs() * ub;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    // let mut rng = StdRng::seed_from_u64(0);
    let mut rng = rand::rng();
    let mut ts = vec![0.; avg_sigmaz.len()];
    let alpha = 0.95;

    for i in 0..num_tries {
        bar.inc(1);
        let mut system = systems::qubitnewfeedbackv7::QubitNewFeedbackV7::new(
            h, l, f0, y0, y1, lb, ub, &mut rng, alpha,
        );
        let tf = system.tf;

        let mut solver = StochasticSolver::new(&mut system, 0.0, rho1, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(to_bloch_unchecked)
            .collect::<Vec<BlochVector>>();

        let mut trajectory = plotpy::Curve::new();
        trajectory
            .set_line_width(0.5)
            .set_line_color(colors[i as usize])
            .draw_3d(
                &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>(),
                &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>(),
                &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>(),
            );

        plot1.add(&trajectory);

        let last = obsv.last().unwrap();
        let mut end = plotpy::Curve::new();

        end.set_line_color(colors[i as usize])
            .set_marker_style("o")
            .set_marker_size(10.0)
            .points_3d_begin()
            .points_3d_add(last[0], last[1], last[2])
            .points_3d_end();

        plot1.add(&end);

        let mut zaxis = plotpy::Curve::new();
        zaxis
            .set_line_color(colors[i as usize])
            .draw(t_out, &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>());
        plot2.set_subplot(1, 3, 1).add(&zaxis);

        let mut newcorr = vec![0.; dy_out.len()];
        let mut acc = 0.;
        let mut t = 0.;
        for i in 0..dy_out.len() - 1 {
            t += dt;
            acc += dy_out[i];
            newcorr[i + 1] = if (acc / t).abs() > ube
                || (acc / t - y1).abs() < lbe
                || (acc / t - y0).abs() > (acc / t - y1).abs()
                || t < tf
            {
                0.
            } else {
                // acc / t - y1
                -(acc / t - y1).abs()
            };
        }

        let mut newcorrs = plotpy::Curve::new();
        newcorrs
            .set_line_color(colors[i as usize])
            .draw(t_out, &newcorr);
        plot2.set_subplot(1, 3, 2).add(&newcorrs);

        if newcorr[dy_out.len() - 1] == 0. {
            converged_traj += 1;
            avg_sigmaz = avg_sigmaz
                .iter()
                .zip(&obsv.iter().map(|o| o[2]).collect::<Vec<f64>>())
                .map(|(x, y)| x + y)
                .collect::<Vec<f64>>();
        }
        ts = t_out.to_vec();
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

    plot1.add(&start);

    let mut sigmaz_avg_traj = plotpy::Curve::new();
    sigmaz_avg_traj
        .set_label(&format!(
            "Number of converged trajectories: {converged_traj}"
        ))
        .draw(
            &ts,
            &avg_sigmaz
                .iter()
                .map(|x| x / (converged_traj as f64))
                .collect::<Vec<f64>>(),
        );

    plot2.set_subplot(1, 3, 3).add(&sigmaz_avg_traj);
    // plot2.set_subplot(1, 3, 1).set_yrange(-1., 1.);
    // plot2.set_subplot(1, 3, 3).set_yrange(-1., 1.);

    println!("Converged trajectories: {converged_traj}");
    // plot1.set_equal_axes(true).show("tempimages")?;
    plot2.show("tempimages")?;

    Ok(())
}
