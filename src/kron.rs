use crate::plots::constrainedlayout;
use crate::solver::Rk4;
use crate::systems;
use crate::utils::*;
use nalgebra as na;
use rand_distr::num_traits::ToPrimitive;
use statrs::statistics::Statistics;

pub fn vectorisationexample() -> SolverResult<()> {
    let d = 3.;
    let h0 =
        na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast::<na::Complex<f64>>();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();

    let h = h0 + hc + f0;
    let l =
        na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast::<na::Complex<f64>>();
    let f1 = na::Matrix3::zeros();

    let lindbladsymm = veclindblad(&na::Matrix3::zeros(), &l.scale(f64::sqrt(2.)));

    let num_tries = 100;
    let final_time = 5.;
    let dt = 0.001;

    let eigs = lindbladsymm.symmetric_eigenvalues();
    let maxeig = eigs.max();
    let spgap = Statistics::max(
        eigs.into_iter()
            .filter(|e| e < &&maxeig)
            .map(|e| *e)
            .collect::<Vec<f64>>(),
    );

    let mut plot = plotpy::Plot::new();
    let maxchi20 = d - 1.;

    for i in 0..num_tries {
        let x0: na::Vector3<na::Complex<f64>> = random_unit_complex_vector();
        let x0 = x0 * x0.adjoint();

        let system = systems::wisemanfme::WisemanFME::new(h, l, f1);
        let mut solver = Rk4::new(system, 0.0, x0, final_time, dt);
        solver.integrate()?;

        let (t_out, rho_out) = solver.results().get();

        let mut traceline = plotpy::Curve::new();
        traceline.draw(
            t_out,
            &rho_out
                .iter()
                .map(|rho| {
                    rho.symmetric_eigenvalues()
                        .map(|eig| (eig - 1. / d).abs())
                        .sum()
                        .powi(2)
                })
                .collect(),
        );

        plot.add(&traceline);
    }

    let num_steps = ((final_time / dt).ceil()).to_usize().unwrap();
    let t_out = (0..num_steps)
        .map(|n| (n as f64) * dt)
        .collect::<Vec<f64>>();

    let mut expline = plotpy::Curve::new();
    expline
        .set_line_color("#FF04FF")
        .set_label(&format!("Spectral gap: {spgap}"))
        .draw(
            &t_out,
            &t_out.iter().map(|t| maxchi20 * (t * spgap).exp()).collect(),
        );

    let mut expline2 = plotpy::Curve::new();
    expline2
        .set_line_color("#DA7422")
        .set_label(&format!("Spectral gap: {}", spgap / 2.))
        .draw(
            &t_out,
            &t_out
                .iter()
                .map(|t| maxchi20 * (t * spgap / 2.).exp())
                .collect(),
        );

    plot.add(&expline).add(&expline2).legend();
    plot.show("tempimage.png")?;
    Ok(())
}
