use core::f64;

use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug)]
/// New feedback with both F0 and F1 for qubits
pub struct QubitFeedback<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
    lastset: LastSet,
    y: f64,
    y1: f64,
    y2: f64,
    ub1: f64,
    ub2: f64,
    tf: f64,
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
        beta: f64,
        epsilon: f64,
        rng: &'a mut R,
    ) -> Self {
        let normal = Normal::standard();
        let tf = (normal.inverse_cdf((beta + 1.) / 2.) / epsilon).powi(2);
        let hhat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            hhat,
            lhat,
            lastset: LastSet::NotSet,
            y: 0.,
            y1,
            y2,
            rng,
            ub1: (y1 - y2).abs() * gamma,
            ub2: (y1 - y2).abs() * gamma / 2.,
            tf,
            wiener: wiener::Wiener::new(),
        }
    }
    pub fn tf(&self) -> f64 {
        self.tf
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitFeedback<'a, R> {
    fn system(&mut self, t: f64, dt: f64, rho: &QubitState, drho: &mut QubitState, dw: &Vec<f64>) {
        let miny = self.y1.min(self.y2);
        let maxy = self.y1.max(self.y2);
        let avg = (if t > self.tf { self.y / t } else { 0. })
            .max(miny)
            .min(maxy);

        let corr = if (avg - self.y2).abs() >= self.ub1 {
            0.
        } else if (avg - self.y2).abs() <= self.ub2 {
            1.
        } else if self.lastset == LastSet::NotSet || self.lastset == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.lastset = if (avg - self.y2).abs() >= self.ub1 {
            LastSet::LeGammaSet
        } else if (avg - self.y2).abs() <= self.ub2 {
            LastSet::GeGamma2Set
        } else {
            self.lastset
        };

        let hhat = self.hhat + self.f0.scale(corr);
        *drho = rouchonstep(dt, &rho, &hhat, &self.lhat, dw[0]);
        let dy = self.measurement(rho, dt, dw[0]);
        self.y += dy;
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}

#[derive(Debug)]
/// New feedback with both F0 and F1 for qubits
pub struct QubitFeedback2<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    hhat: QubitOperator,
    lhat: QubitOperator,
    lastset: LastSet,
    y: f64,
    y1: f64,
    y2: f64,
    lastdy: usize,
    lastdys: Vec<f64>,
    ub1: f64,
    ub2: f64,
    tf: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitFeedback2<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        hc: QubitOperator,
        f0: QubitOperator,
        f1: QubitOperator,
        y1: f64,
        y2: f64,
        k: usize,
        gamma: f64,
        beta: f64,
        epsilon: f64,
        rng: &'a mut R,
    ) -> Self {
        let normal = Normal::standard();
        let tf = (normal.inverse_cdf((beta + 1.) / 2.) / epsilon).powi(2);
        let hhat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            hhat,
            lhat,
            lastset: LastSet::NotSet,
            y: 0.,
            y1,
            y2,
            lastdy: 0,
            lastdys: vec![0.; k],
            rng,
            ub1: (y1 - y2).abs() * gamma,
            ub2: (y1 - y2).abs() * gamma / 2.,
            tf,
            wiener: wiener::Wiener::new(),
        }
    }
    pub fn tf(&self) -> f64 {
        self.tf
    }
}

impl<'a, R: wiener::Rng + ?Sized> StochasticSystem<QubitState> for QubitFeedback2<'a, R> {
    fn system(&mut self, t: f64, dt: f64, rho: &QubitState, drho: &mut QubitState, dw: &Vec<f64>) {
        let miny = self.y1.min(self.y2);
        let maxy = self.y1.max(self.y2);
        let avg = (if t > self.tf {
            self.y / (self.lastdys.len() as f64 * dt)
        } else {
            0.
        })
        .max(miny)
        .min(maxy);

        let corr = if (avg - self.y2).abs() >= self.ub1 {
            0.
        } else if (avg - self.y2).abs() <= self.ub2 {
            1.
        } else if self.lastset == LastSet::NotSet || self.lastset == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.lastset = if (avg - self.y2).abs() >= self.ub1 {
            LastSet::LeGammaSet
        } else if (avg - self.y2).abs() <= self.ub2 {
            LastSet::GeGamma2Set
        } else {
            self.lastset
        };

        let hhat = self.hhat + self.f0.scale(corr);
        *drho = rouchonstep(dt, &rho, &hhat, &self.lhat, dw[0]);
        let dy = self.measurement(rho, dt, dw[0]);
        self.y += dy - self.lastdys[self.lastdy];
        self.lastdys[self.lastdy] = dy;
        self.lastdy = (self.lastdy + 1) % self.lastdys.len();
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
