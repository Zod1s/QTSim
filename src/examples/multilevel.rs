use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn multilevel() -> SolverResult<()> {
    // let h = PAULI_Z;
    // let l = PAULI_Z;
    // let f = QubitOperator::zeros();
    // let l = PAULI_X;
    // let f = PAULI_Y;

    let h = na::Matrix4::from_diagonal(&na::Vector4::new(1.0, 2.0, 3.0, 4.0)).cast();
    // let l = na::Matrix4::from_diagonal(&na::Vector4::new(1.0, 2.0, 3.0, 4.0)).cast();
    // let f = Operator::<na::Const<4>>::zeros();
    let l = na::matrix![0., 1., 0., 0.; 1., 0., 0., 0.; 0., 0., 0., 1.; 0., 0., 1., 0.].cast();
    let f1 = na::Matrix2::<na::Complex<f64>>::identity()
        .kronecker(&PAULI_Y)
        .scale(-1.);
    let hc = na::matrix![0., 0., 0., 0.; 0., 0., 1., 0.; 0., 1., 0., 0.; 0., 0., 0., 0.].cast();
    let meas = na::Matrix4::from_diagonal(&na::Vector4::new(2., 1., -1., -2.)).cast();

    let dt = 0.0001;
    let decimation = 10;
    let final_time = 10.0;
    let num_tries = 10;
    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff00ff", "#e1ff00",
    ];

    let mut rng = rand::rng();

    let mut plot = plotpy::Plot::new();

    let bar = ProgressBar::new(num_tries).with_style(
        ProgressStyle::default_bar()
            .template("Simulating: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
            .unwrap(),
    );
    for i in 0..num_tries {
        bar.inc(1);
        // let x0 = random_pure_state();
        // let x0 = random_qubit_state();
        // let x0 = na::Matrix2::new(1., 0., 0., 0.).cast();

        // let mut solver = Rk4::new(system, 0.0, x0, 20.0, 0.001);
        let x0 = random_unit_complex_vector::<4>();
        let x0 = x0 * x0.conjugate().transpose();
        // let system = systems::wisemanfme::WisemanFME::new(h + hc, l, f1);
        // let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
        let mut system = systems::wisemansse::WisemanSSE::new(h + hc, l, f1, &mut rng);
        let mut solver = StochasticSolver::new(&mut system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        // let (t_out, rho_out) = solver.results().get();
        let (t_out, rho_out, _) = solver.results().get();

        let obsv = rho_out
            .iter()
            .map(|rho| (meas * rho).trace().re)
            .collect::<Vec<f64>>();

        let mut zobsv = plotpy::Curve::new();
        let t_out_plot = (0..t_out.len() / decimation)
            .map(|i| t_out[i * decimation])
            .collect::<Vec<f64>>();
        let obsv_plot = (0..obsv.len() / decimation)
            .map(|i| obsv[i * decimation])
            .collect();
        zobsv
            .set_line_color(colors[i as usize])
            .draw(&t_out_plot, &obsv_plot);

        plot.add(&zobsv);
    }
    bar.finish();
    println!("Done simulations");

    plot.show("tempimage")?;

    Ok(())
}
