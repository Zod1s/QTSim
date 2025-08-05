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
    let l = PAULI_Z;
    let f = QubitOperator::zeros();
    // let l = PAULI_X;
    // let f = PAULI_Y;
    let system = systems::qubitwisemanfme::QubitWisemanFME::new(h, l, f);

    let mut plot = plotpy::Plot::new();
    plots::plot_bloch_sphere(&mut plot);

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff00ff", "#e1ff00",
    ];

    for i in 0..10 {
        // let x0 = random_pure_state();
        let x0 = random_qubit_state();
        // let x0 = na::Matrix2::new(1., 0., 0., 0.).cast();

        let mut solver = Rk4::new(system, 0.0, x0, 20.0, 0.001);
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

    plot.set_equal_axes(true).show("tempimages")?;

    Ok(())
}
