// naive discretisation of the system, to check whether normalisation is preserved
use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
pub struct QubitNewFeedbackV4<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    f1: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
    y: f64,
    yd: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitNewFeedbackV4<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        f0: QubitOperator,
        f1: QubitOperator,
        rhod: QubitState,
        rng: &'a mut R,
    ) -> Self {
        let hhat = h + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;
        let yd = ((l + l.adjoint()) * rhod).trace().re;

        Self {
            h,
            l,
            f0,
            f1,
            hhat,
            lhat,
            y: 0.,
            yd,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitNewFeedbackV4<'a, R> {
    fn system(&mut self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let corr = if t > 0. { self.y / t - self.yd } else { 0. };
        let corr = corr.powi(2);
        let hhat = self.hhat + self.f0.scale(corr);
        let drho = (hamiltonian_term(&hhat, x) + measurement_term(&self.lhat, x)).scale(dt)
            + noise_term(&self.lhat, x).scale(dw[0]);
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
