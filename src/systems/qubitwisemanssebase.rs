use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
pub struct QubitWisemanSSE<'a, R: wiener::Rng + ?Sized> {
    l: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitWisemanSSE<'a, R> {
    pub fn new(h: QubitOperator, l: QubitOperator, f: QubitOperator, rng: &'a mut R) -> Self {
        let lhat = l - f * na::Complex::I;
        let hhat = h + (f * l + l.adjoint() * f).scale(0.5);

        Self {
            l,
            hhat,
            lhat,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitWisemanSSE<'a, R> {
    fn system(&mut self, _: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let drho = (hamiltonian_term(&self.hhat, x) + measurement_term(&self.lhat, x)).scale(dt)
            + noise_term(&self.lhat, x).scale(dw[0]);
        *dx = drho - QubitOperator::identity().scale(drho.trace().re)
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
