use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug)]
/// New feedback with both F0 and F1 for qubits
pub struct QubitFeedback<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
    y: f64,
    y0: f64,
    y1: f64,
    lb: f64,
    ub: f64,
    tf: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitFeedback<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        hc: QubitOperator,
        f0: QubitOperator,
        f1: QubitOperator,
        y0: f64,
        y1: f64,
        lb: f64,
        ub: f64,
        alpha: f64,
        rng: &'a mut R,
    ) -> Self {
        let normal = Normal::standard();
        let epsilon = (y0 - y1).abs() / 4.;
        let tf = (normal.cdf((alpha + 1.) / 2.) / epsilon).powi(2);
        // let tf = 0.;
        let hhat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            hhat,
            lhat,
            y: 0.,
            y0,
            y1,
            rng,
            lb: (y0 - y1).abs() * lb,
            ub: (y0 - y1).abs() * ub,
            tf,
            wiener: wiener::Wiener::new(),
        }
    }
    pub fn tf(&self) -> f64 {
        self.tf
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitFeedback<'a, R> {
    fn system(&mut self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let avg = if t > self.tf { self.y / t } else { 0. };
        let corr =
            if avg.abs() > self.ub || (avg - self.y0).abs() > (avg - self.y1).abs() || t <= self.tf
            {
                0.
            } else if (avg - self.y0).abs() < self.lb {
                avg - self.y1
                // -(avg - self.y1).abs()
            } else {
                0.
            };

        let id = QubitOperator::identity();
        let hhat = self.hhat + self.f0.scale(corr);
        let fst = (hhat * na::Complex::I + self.lhat.adjoint() * self.lhat.scale(0.5)).scale(dt);
        let snd = self
            .lhat
            .scale((self.lhat * x + x * self.lhat.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.lhat * self.lhat).scale(dw[0].powi(2) - dt).scale(0.5);

        let m = id - fst + snd + thd;

        let num = m * x * m.adjoint();
        *dx = num.scale(1. / num.trace().re) - x;
        let dy = self.measurement(x, dt, dw[0]);
        self.y += dy;
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
