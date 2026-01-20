use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
/// Nonphysical evolution with H_c and F1 for a qubit using Rouchon integration
pub struct QubitNotPhysicalRouchon<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f: QubitOperator,
    dy: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitNotPhysicalRouchon<'a, R> {
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

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<'a, R, QubitState>
    for QubitNotPhysicalRouchon<'a, R>
{
    fn system(&mut self, t: f64, dt: f64, rho: &QubitState, drho: &mut QubitState, dw: &Vec<f64>) {
        let hhat = self.h + self.f.scale(self.dy / dt);
        *drho = rouchonstep(dt, &rho, &hhat, &self.l, dw[0]);

        self.dy = self.measurement(rho, dt, dw[0]);
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }

    fn setrng(&mut self, rng: &'a mut R) {
        self.rng = rng;
    }
}
