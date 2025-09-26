use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
/// Rouchon discretisation of the Wiseman-Milburn feedback for qubits with control applied after
/// evolution
pub struct QubitSequentialControl<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f: QubitOperator,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitSequentialControl<'a, R> {
    pub fn new(h: QubitOperator, l: QubitOperator, f: QubitOperator, rng: &'a mut R) -> Self {
        Self {
            h,
            l,
            f,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitSequentialControl<'a, R> {
    fn system(&mut self, _: f64, dt: f64, rho: &QubitState, drho: &mut QubitState, dw: &Vec<f64>) {
        // let id = QubitOperator::identity();
        // let fst = (self.h * na::Complex::I + self.l.adjoint() * self.l.scale(0.5)).scale(dt);
        // let snd = self
        //     .l
        //     .scale((self.l * x + x * self.l.adjoint()).trace().re * dt + dw[0]);
        // let thd = (self.l * self.l).scale(dw[0].powi(2) - dt).scale(0.5);
        //
        // let m = id - fst + snd + thd;
        //
        // let num = m * x * m.adjoint();
        // let rho = num.scale(1. / num.trace().re);
        let sigma = rho + rouchonstep(dt, &rho, &self.h, &self.l, dw[0]);
        let dy = self.measurement(rho, dt, dw[0]);

        let feedback = |tau| -commutator(&self.f, tau) * na::Complex::I;

        // I don't know if I am doing this right, it may lose the positivity constraint
        // I am evolving the state for dt, then applying the series expansion of exp(Mdy)
        // to the new state, but I don't know if it makes sense to stop to the second order
        // from a numerical point of view
        // Also, I need to explore whether I need to use dt or dy^2 for the actual dy^2

        let feedrho = feedback(&sigma);

        *drho = sigma - rho + (feedrho).scale(dy) + (feedback(&feedrho)).scale(0.5 * dy.powi(2));
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
