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
    fn system(&self, _: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let id = QubitOperator::identity();
        let fst =
            (self.hhat * na::Complex::I + self.lhat.adjoint() * self.lhat.scale(0.5)).scale(dt);
        let snd = self
            .lhat
            .scale((self.lhat * x + x * self.lhat.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.lhat * self.lhat).scale(dw[0].powi(2) - dt);

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

#[derive(Debug)]
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
    fn system(&self, _: f64, dt: f64, x: &QubitState, dx: &mut QubitState, dw: &Vec<f64>) {
        let id = QubitOperator::identity();
        let fst = (self.h * na::Complex::I + self.l.adjoint() * self.l.scale(0.5)).scale(dt);
        let snd = self
            .l
            .scale((self.l * x + x * self.l.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.l * self.l).scale(dw[0].powi(2) - dt);

        let m = id - fst + snd + thd;

        let num = m * x * m.adjoint();
        let rho = num.scale(1. / num.trace().re);
        let dy = self.measurement(x, dt, dw[0]);

        let feedback = |sigma| -commutator(&self.f, sigma) * na::Complex::I;

        // I don't know if I am doing this right, it may lose the positivity constraint
        // I am evolving the state for dt, then applying the series expansion of exp(Mdy)
        // to the new state, but I don't know if it makes sense to stop to the second order
        // from a numerical point of view
        // Also, I need to explore whether I need to use dt or dy^2 for the actual dy^2

        let feedrho = feedback(&rho);

        *dx = rho - x + (feedrho).scale(dy) + (feedback(&feedrho)).scale(0.5 * dy.powi(2));
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &QubitState, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}

#[derive(Debug)]
pub struct QubitNewFeedback<'a, R: wiener::Rng + ?Sized> {
    h: QubitOperator,
    l: QubitOperator,
    f0: QubitOperator,
    f1: QubitOperator,
    hcor: QubitOperator,
    lhat: QubitOperator,
    y: f64,
    yd: f64,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized> QubitNewFeedback<'a, R> {
    pub fn new(
        h: QubitOperator,
        l: QubitOperator,
        f0: QubitOperator,
        f1: QubitOperator,
        rhod: QubitState,
        rng: &'a mut R,
    ) -> Self {
        let lhat = l - f1 * na::Complex::I;
        let hcor = (f1 * l + l.adjoint() * f1).scale(0.5);
        let yd = ((l + l.adjoint()) * rhod).trace().re;

        Self {
            h,
            l,
            f0,
            f1,
            hcor,
            lhat,
            y: 0.,
            yd,
            rng,
            wiener: wiener::Wiener::new(),
        }
    }
}
