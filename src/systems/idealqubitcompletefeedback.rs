use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
/// New feedback with both F0 and F1 for qubits, using the ideal yt
pub struct QubitFeedback<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
    lastset: LastSet,
    y1: f64,
    y2: f64,
    ub1: f64,
    ub2: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitFeedback<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        hc: QubitOperator,
        f0: QubitOperator,
        f1: QubitOperator,
        y1: f64,
        y2: f64,
        gamma: f64,
        rng: &'a mut R,
    ) -> Self {
        let hhat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            hhat,
            lhat,
            lastset: LastSet::NotSet,
            y1,
            y2,
            rng,
            ub1: (y1 - y2).abs() * gamma,
            ub2: (y1 - y2).abs() * gamma / 2.,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitFeedback<'a, R> {
    fn system(&mut self, t: f64, dt: f64, rho: &QubitState, drho: &mut QubitState, dw: &Vec<f64>) {
        let y = ((self.l + self.l.adjoint()) * rho).trace().re;

        let corr = if (y - self.y2).abs() >= self.ub1 {
            0.
        } else if (y - self.y2).abs() <= self.ub2 {
            1.
        } else if self.lastset == LastSet::NotSet || self.lastset == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.lastset = if (y - self.y2).abs() >= self.ub1 {
            LastSet::LeGammaSet
        } else if (y - self.y2).abs() <= self.ub2 {
            LastSet::GeGamma2Set
        } else {
            self.lastset
        };

        let hhat = self.hhat + self.f0.scale(corr);
        *drho = rouchonstep(dt, &rho, &hhat, &self.lhat, dw[0]);
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
