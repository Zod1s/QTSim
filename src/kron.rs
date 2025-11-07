use core::f64;

use crate::lyapunov;
use crate::plots::constrainedlayout;
use crate::solver::Rk4;
use crate::systems;
use crate::utils::*;
use nalgebra as na;
use rand_distr::num_traits::ToPrimitive;
use statrs::statistics::Statistics;

pub fn vectorisationexample() -> SolverResult<()> {
    let d: f64 = 3.;
    let h0 = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 0.; 1., 0., 1.; 0., 1., 0.].cast();

    let h = h0 + hc + f0;
    let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let f1 = na::Matrix3::zeros();

    let lindbladsymm = veclindblad(&na::Matrix3::zeros(), &l.scale(f64::sqrt(2.)));

    let num_tries = 200;
    let final_time = 5.;
    let dt = 0.001;

    let eigs = lindbladsymm.symmetric_eigenvalues();
    let maxeig = eigs.max();
    let mut spgap = -f64::INFINITY;
    for eig in eigs.iter() {
        if eig < &maxeig && &spgap < eig {
            spgap = *eig;
        }
    }
    spgap -= maxeig;

    let mut plot = plotpy::Plot::new();
    let maxchi20 = d - 1.;

    for i in 0..num_tries {
        // let x0 = random_pure_state::<na::U3>();
        let x0 = na::Matrix3::from_diagonal(&na::Vector3::new(0., 0., 1.)).cast();

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

pub fn lyapunovtrend() -> SolverResult<()> {
    let d: f64 = 3.;
    let h0 = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 0.; 1., 0., 1.; 0., 1., 0.].cast();

    let h = h0 + hc + f0;
    let l = na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast();
    let f1 = na::Matrix3::zeros();

    let num_tries = 200;
    let final_time = 5.;
    let dt = 0.001;

    let lmin: f64 = 0.0634;
    let lmax: f64 = 1.8612;

    let mut plot = plotpy::Plot::new();
    // plot.set_log_y(true);

    for i in 0..num_tries {
        // let x0 = random_pure_state::<na::U3>();
        let x0 = na::Matrix3::from_diagonal(&na::Vector3::new(0., 0., 1.)).cast();

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
                    GELLMANNMATRICES
                        .iter()
                        .map(|matrix| (rho * matrix).trace().re.powi(2))
                        .sum::<f64>()
                        .sqrt()
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
        .set_label(&format!("Quadratic"))
        .draw(
            &t_out,
            &t_out
                .iter()
                .map(|t| ((d - 1.) / d * lmax / lmin).sqrt() * (-t / (2. * lmax)).exp())
                .collect(),
        );

    plot.add(&expline).legend();
    plot.show("tempimage.png")?;

    Ok(())
}

// let b = GELLMANNMATRICES
//     .iter()
//     .map(|matrix| (matrix * l_hat).trace().re.powi(2))
//     .sum();
