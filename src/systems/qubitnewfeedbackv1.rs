use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
pub struct QubitNewFeedbackV1<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    f1: QubitOperator,
    hcor: QubitOperator,
    lhat: QubitOperator,
    y: f64,
    yd: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitNewFeedbackV1<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        f0: QubitOperator,
        f1: QubitOperator,
        rhod: QubitState,
        rng: &'a mut R,
    ) -> Self {
        let lhat = l - f1 * na::Complex::I;
        let hcor = (f1 * l + l.adjoint() * f1).scale(0.5);
        let yd = ((l + l.adjoint()) * rhod).trace().re;

        Self {
            h,
            l,
            f0,
            f1,
            hcor,
            lhat,
            y: 0.,
            yd,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitNewFeedbackV1<'a, R> {
    fn system(&mut self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let id = QubitOperator::identity();
        let corr = if t > 0. { self.y / t - self.yd } else { 0. };
        // let corr = corr.powi(2);
        let hhat = self.h + self.hcor + self.f0.scale(corr);
        let fst = (hhat * na::Complex::I + self.lhat.adjoint() * self.lhat.scale(0.5)).scale(dt);
        let snd = self
            .lhat
            .scale((self.lhat * x + x * self.lhat.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.lhat * self.lhat).scale(dw[0].powi(2) - dt);

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
