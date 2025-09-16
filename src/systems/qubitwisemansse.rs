use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
/// Rouchon discretisation of the SDE with Wiseman-Milburn feedback for qubits
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
        let id = QubitOperator::identity();
        let fst =
            (self.hhat * na::Complex::I + self.lhat.adjoint() * self.lhat.scale(0.5)).scale(dt);
        let snd = self
            .lhat
            .scale((self.lhat * x + x * self.lhat.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.lhat * self.lhat).scale(dw[0].powi(2) - dt).scale(0.5);

        let m = id - fst + snd + thd;

        let num = m * x * m.adjoint();
        *dx = num.scale(1. / num.trace().re) - x;
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
