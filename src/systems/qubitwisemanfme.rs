use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Clone, Copy, Debug)]
pub struct QubitWisemanFME {
    hhat: QubitOperator,
    lhat: QubitOperator,
}

impl QubitWisemanFME {
    pub fn new(h: QubitOperator, l: QubitOperator, f: QubitOperator) -> Self {
        let lhat = l - f * na::Complex::I;
        let hhat = h + (f * l + l.adjoint() * f).scale(0.5);
        Self { hhat, lhat }
    }
}

impl System<QubitState> for QubitWisemanFME {
    fn system(&self, _: f64, rho: &QubitState, drho: &mut QubitState) {
        *drho = -commutator(&self.hhat, rho) * na::Complex::I
            + self.lhat * rho * self.lhat.adjoint()
            - anticommutator(&(self.lhat.adjoint() * self.lhat), rho).scale(0.5);
    }
}
