// naive discretisation of the system, to check whether normalisation is preserved
use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
pub struct QubitNewFeedbackV5<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    y: f64,
    y0: f64,
    y1: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitNewFeedbackV5<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        f0: QubitOperator,
        y0: f64,
        y1: f64,
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
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitNewFeedbackV5<'a, R> {
    fn system(&mut self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let acc = if t > 0. { self.y / t } else { 0. };
        let corr = if (acc - self.y0).abs() > (acc - self.y1).abs() {
            0.
        } else {
            -1. // self.y0 - self.y1
        };

        let hhat = self.h + self.f0.scale(corr);
        let drho = (hamiltonian_term(&hhat, x) + measurement_term(&self.l, x)).scale(dt)
            + noise_term(&self.l, x).scale(dw[0]);
        *dx = drho - QubitOperator::identity() * drho.trace();
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
