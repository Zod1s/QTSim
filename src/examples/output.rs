use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn output() -> SolverResult<()> {
    // let h0 = 1.;
    // let h1 = -1.;
    // let h = na::Matrix2::new(h0, 0., 0., h1).cast();
    // let l0 = 1.;
    // let l1 = -1.;
    // let l = na::Matrix2::new(l0, 0., 0., l1).cast();
    let h = PAULI_Z;
    let l = PAULI_Z;
    let f = QubitOperator::zeros();

    // let mut rng = StdRng::seed_from_u64(0);
    let mut rng = rand::rng();
    let mut system = systems::qubitwisemansse::QubitWisemanSSE::new(h, l, f, &mut rng);
    // let x0 = random_qubit_state();
    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 8.0;
    let dt = 0.001;

    let rho0 = na::Matrix2::new(1.0, 0.0, 0.0, 0.0).cast();
    let rho1 = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    let y0 = ((l + l.adjoint()) * rho0).trace().re; // output we would have at the equilibrium
    let y1 = ((l + l.adjoint()) * rho1).trace().re; // output we would have at the equilibrium

    let mut plot1 = plotpy::Plot::new();
    // let mut plot2 = plotpy::Plot::new();
    plots::plot_bloch_sphere(plot1.set_subplot_3d(1, 2, 1))?;

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );
    let mut zerocount = 0;

    for i in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out, dy_out) = solver.results().get();

        // let mut y_out = vec![0.; dy_out.len()];
        // let mut corr = vec![0.; dy_out.len()];
        let mut newcorr = vec![0.; dy_out.len()];
        let mut acc = 0.;
        let mut t = 0.;
        for i in 0..dy_out.len() - 1 {
            t += dt;
            acc += dy_out[i];
            // y_out[i + 1] = acc;
            // corr[i + 1] = acc / t - y1;
            newcorr[i + 1] = if (acc / t - y0).abs() > (acc / t - y1).abs() {
                0.
            } else {
                y0 - y1
            };
        }

        // let mut y = plotpy::Curve::new();
        // y.set_line_color(colors[i as usize]).draw(t_out, &y_out);
        // plot1.add(&y);
        //
        // let mut corrs = plotpy::Curve::new();
        // corrs.set_line_color(colors[i as usize]).draw(t_out, &corr);
        // plot2.add(&corrs);
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

        let last = obsv.last().unwrap();
        let mut end = plotpy::Curve::new();

        end.set_line_color(colors[i as usize]) //(colors[i as usize])
            .set_marker_style("o")
            .set_marker_size(10.0)
            .points_3d_begin()
            .points_3d_add(last[0], last[1], last[2])
            .points_3d_end();

        plot1.set_subplot_3d(1, 2, 1).add(&end);
        plot1.set_subplot_3d(1, 2, 1).add(&trajectory);

        let mut newcorrs = plotpy::Curve::new();
        newcorrs
            .set_line_color(colors[i as usize])
            .draw(t_out, &newcorr);
        plot1.set_subplot(1, 2, 2).add(&newcorrs);
        zerocount += if newcorr[dy_out.len() - 1] == 0. {
            1
        } else {
            0
        };
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

    plot1.set_subplot_3d(1, 2, 1).add(&start);
    plot1.set_subplot_3d(1, 2, 1).set_equal_axes(true);
    plot1.show("tempimages")?;
    println!("Number of zero trajectories: {zerocount}");

    Ok(())
}
