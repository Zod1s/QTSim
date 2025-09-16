use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
/// Nonphysical evolution with H_c and F1 for a qubit using Euler-Maruyama integration
pub struct QubitNotPhysical<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f: QubitOperator,
    dy: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitNotPhysical<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        hc: QubitOperator,
        f: QubitOperator,
        rng: &'a mut R,
    ) -> Self {
        Self {
            h: h + hc,
            l,
            f,
            dy: 0.,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitNotPhysical<'a, R> {
    fn system(&mut self, t: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let drho = (hamiltonian_term(&(self.h + self.f.scale(self.dy / dt)), x)
            + measurement_term(&self.l, x)
            + measurement_term(&self.f, x))
        .scale(dt)
            + noise_term(&self.l, x).scale(dw[0]);
        *dx = drho - QubitOperator::identity() * drho.trace();
        self.dy = self.measurement(x, dt, dw[0]);
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
