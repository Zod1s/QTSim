use crate::plots;
use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn output() -> SolverResult<()> {
    // let h = PAULI_Z;
    // let l = PAULI_X;
    let f = PAULI_Y;
    let h = PAULI_Z + PAULI_Y;
    let l = PAULI_X + PAULI_Z;
    // let f = QubitOperator::new(
    //     na::Complex::ZERO,
    //     -na::Complex::I,
    //     na::Complex::I,
    //     na::Complex::new(2., 0.),
    // );

    let mut rng = StdRng::seed_from_u64(0);
    let mut system = systems::qubitwisemansse::QubitWisemanSSE::new(h, l, f, &mut rng);
    // let x0 = random_qubit_state();
    let x0 = na::Matrix2::new(0.5, 0.5, 0.5, 0.5).cast();
    let x0bloch = to_bloch(&x0)?;

    let num_tries = 10;
    let final_time: f64 = 8.0;
    let dt = 0.001;

    let rhod = na::Matrix2::new(0.0, 0.0, 0.0, 1.0).cast();
    let yd = ((l + l.adjoint()) * rhod).trace().re; // output we would have at the equilibrium

    let mut plot1 = plotpy::Plot::new();
    let mut plot2 = plotpy::Plot::new();
    let mut plot3 = plotpy::Plot::new();

    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff0037", "#e1ff00",
    ];

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );

    for i in 0..num_tries {
        bar.inc(1);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, _, dy_out) = solver.results().get();

        let mut y_out = vec![0.; dy_out.len()];
        let mut corr = vec![0.; dy_out.len()];
        let mut acc = 0.;
        let mut y_part = vec![0.; dy_out.len()];
        let delta = 50;
        let mut last_ys = vec![0.; delta];
        let deltat = delta as f64 * dt;
        let mut index = 0;
        for i in 0..dy_out.len() - 1 {
            acc += dy_out[i];
            y_out[i + 1] = acc;
            corr[i + 1] = (acc / ((i + 1) as f64 * dt)) - yd;
            y_part[i + 1] = y_part[i] - last_ys[index] + dy_out[i] / deltat;
            last_ys[index] = dy_out[i] / deltat;
            index += 1;
            index = index % delta;
        }

        let mut y = plotpy::Curve::new();
        y.set_line_color(colors[i as usize]).draw(t_out, &y_out);
        plot1.add(&y);

        let mut corrs = plotpy::Curve::new();
        corrs.set_line_color(colors[i as usize]).draw(t_out, &corr);
        plot2.add(&corrs);

        let mut party = plotpy::Curve::new();
        party
            .set_line_color(colors[i as usize])
            .draw(t_out, &y_part);
        plot3.add(&party);
    }
    bar.finish();

    plot1.show("tempimages")?;
    plot2.show("tempimages")?;
    plot3.show("tempimages")?;

    Ok(())
}
