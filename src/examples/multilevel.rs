use crate::solver::{Rk4, StochasticSolver};
use crate::systems;
use crate::utils::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::num_traits::ToPrimitive;

pub fn multilevel() -> SolverResult<()> {
    // let h = PAULI_Z;
    // let hc = PAULI_X;
    // let l = PAULI_Z;
    // let f1 = QubitOperator::zeros();

    let h = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let hc = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();
    let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let f1 = Operator::<na::Const<3>>::zeros();
    // let l = na::matrix![0., 1., 0., 0.; 1., 0., 0., 0.; 0., 0., 0., 1.; 0., 0., 1., 0.].cast();
    // let f1 = na::Matrix2::<na::Complex<f64>>::identity()
    //     .kronecker(&PAULI_Y)
    //     .scale(-1.);
    // let hc = na::matrix![0., 0., 0., 0.; 0., 0., 1., 0.; 0., 1., 0., 0.; 0., 0., 0., 0.].cast();
    let meas = na::Matrix3::from_diagonal(&na::Vector3::new(-1., 2., 3.)).cast();

    let dt = 0.0001;
    let decimation = 10;
    let final_time = 20.0;
    let num_tries = 10;
    let colors = [
        "#00FF00", "#358763", "#E78A18", "#00fbff", "#3e00ff", "#e64500", "#ffee00", "#0078ff",
        "#ff00ff", "#e1ff00",
    ];

    let mut rng = rand::rng();

    let mut plot = plotpy::Plot::new();
    let system = systems::wisemanfme::WisemanFME::new(h + hc, l, f1);

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
        let x0 = random_pure_state::<na::U3>(None);
        // let system = systems::wisemanfme::WisemanFME::new(h + hc, l, f1);
        // let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
        // let mut system = systems::wisemansse::WisemanSSE::new(h + hc, l, f1, &mut rng);
        let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        // let (t_out, rho_out) = solver.results().get();
        let (t_out, rho_out) = solver.results().get();

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

        println!(
            "Fidelity {i} = {}",
            fidelity(
                &rho_out[rho_out.len() - 1],
                &Operator::<na::Const<3>>::identity().scale(1. / 3.)
            )
        );
    }
    bar.finish();

    let mut xline = plotpy::Curve::new();
    xline.set_line_color("#000000").draw_ray(
        0.,
        l.trace().re / 3.,
        plotpy::RayEndpoint::Horizontal,
    );
    plot.add(&xline);

    let mut xline2 = plotpy::Curve::new();
    xline2
        .set_line_color("#875699")
        .draw_ray(0., l[(1, 1)].re, plotpy::RayEndpoint::Horizontal);
    plot.add(&xline2);

    println!("Done simulations");

    plot.show("tempimages")?;

    Ok(())
}
