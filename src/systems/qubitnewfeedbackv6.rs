use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
pub struct QubitNewFeedbackV6<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    y: f64,
    y0: f64,
    y1: f64,
    lb: f64,
    ub: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitNewFeedbackV6<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        f0: QubitOperator,
        y0: f64,
        y1: f64,
        lb: f64,
        ub: f64,
        rng: &'a mut R,
    ) -> Self {
        Self {
            h,
            l,
            f0,
            y: 0.,
            y0,
            y1,
            rng,
            wiener: wiener::Wiener::new(),
            lb: (y0 - y1).abs() * lb,
            ub: (y0 - y1).abs() * ub,
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitNewFeedbackV6<'a, R> {
    fn system(&mut self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let avg = if t > 0. { self.y / t } else { 0. };

        // Here we set the correction to zero in these three cases:
        // - we are measuring a too large value, since it may be due to the noise in the
        // initial transient
        // - we are sufficiently close to the equilibrium, not to disturb the convergence
        // - we are closer to the equilibrium value than to the non-equlibrium one
        // let corr = if avg.abs() > self.ub
        //     || (avg - self.y1).abs() < self.lb makes no sense to add this onec we have the
        //     following condition
        //     || (avg - self.y0).abs() > (avg - self.y1).abs()
        let corr = if (avg - self.y0).abs() > (avg - self.y1).abs() {
            0.
        } else {
            avg - self.y1
            // -(avg - self.y1).abs()
        };

        let hhat = self.h + self.f0.scale(corr);

        let id = QubitOperator::identity();
        let fst = (hhat * na::Complex::I + self.l.adjoint() * self.l.scale(0.5)).scale(dt);
        let snd = self
            .l
            .scale((self.l * x + x * self.l.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.l * self.l).scale(dw[0].powi(2) - dt).scale(0.5);

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
