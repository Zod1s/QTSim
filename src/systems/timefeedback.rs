use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug)]
/// New feedback with both F0 and F1 for multilevel systems, using the actual yt
pub struct Controller<
    'a,
    R: wiener::Rng + ?Sized,
    D: na::Dim + na::DimName + Sized + std::marker::Copy,
> where
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    h: Operator<D>,
    l: Operator<D>,
    f0: Operator<D>,
    h_hat: Operator<D>,
    l_hat: Operator<D>,
    y: f64,
    y1: f64,
    lb: f64,
    ub: f64,
    tf: f64,
    y_length: usize,
    window_length: f64,
    last_ys: Vec<f64>,
    index: usize,
    is_active: bool,
    max_active_time: f64,
    last_set: LastSet,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized, D: na::Dim + na::DimName + Sized + std::marker::Copy>
    Controller<'a, R, D>
where
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    pub fn new(
        h: Operator<D>,
        l: Operator<D>,
        hc: Operator<D>,
        f0: Operator<D>,
        f1: Operator<D>,
        y1: f64,
        delta: f64,
        gamma: f64,
        alpha: f64,
        epsilon: f64,
        y_length: usize,
        max_active_time: f64,
        rng: &'a mut R,
    ) -> Self {
        let normal = Normal::standard();
        let tf = (normal.cdf((alpha + 1.) / 2.) / epsilon).powi(2);
        let h_hat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let l_hat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            h_hat,
            l_hat,
            y: 0.,
            y1,
            rng,
            tf,
            y_length,
            window_length: y_length as f64 * dt,
            last_ys: vec![0.; y_length],
            index: 0,
            is_active: false,
            max_active_time,
            last_set: LastSet::NotSet,
            ub: 2. * delta - gamma + y1,
            lb: 2. * delta - 2. * gamma + y1,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized, D: na::Dim + na::DimName + Sized + std::marker::Copy>
    StochasticSystem<State<D>> for Controller<'a, R, D>
where
    D: na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    fn system(&mut self, t: f64, dt: f64, rho: &State<D>, drho: &mut State<D>, dw: &Vec<f64>) {
        let avg = if t > self.tf { self.y / t } else { 0. };

        let corr = if avg <= self.lb {
            0.
        } else if avg >= self.ub {
            1.
        } else if self.last_set == LastSet::NotSet || self.last_set == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.last_set = if avg <= self.lb {
            LastSet::LeGammaSet
        } else if avg >= self.ub {
            LastSet::GeGamma2Set
        } else {
            self.last_set
        };

        let h_hat = self.h_hat + self.f0.scale(corr);
        *drho = rouchonstep(dt, &rho, &h_hat, &self.l_hat, dw[0]);
        let dy = self.measurement(rho, dt, dw[0]);
        self.y += dy;
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &State<D>, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}

#[derive(Debug)]
/// New feedback with both F0 and F1 for multilevel systems, using the actual yt
pub struct Feedback2<
    'a,
    R: wiener::Rng + ?Sized,
    D: na::Dim + na::DimName + Sized + std::marker::Copy,
> where
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    h: Operator<D>,
    l: Operator<D>,
    f0: Operator<D>,
    h_hat: Operator<D>,
    l_hat: Operator<D>,
    y: f64,
    y1: f64,
    lb: f64,
    ub: f64,
    tf: f64,
    lastdy: usize,
    lastdys: Vec<f64>,
    count: usize,
    last_set: LastSet,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized, D: na::Dim + na::DimName + Sized + std::marker::Copy>
    Feedback2<'a, R, D>
where
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    pub fn new(
        h: Operator<D>,
        l: Operator<D>,
        hc: Operator<D>,
        f0: Operator<D>,
        f1: Operator<D>,
        y1: f64,
        k: usize,
        delta: f64,
        gamma: f64,
        alpha: f64,
        epsilon: f64,
        rng: &'a mut R,
    ) -> Self {
        let normal = Normal::standard();
        let tf = (normal.cdf((alpha + 1.) / 2.) / epsilon).powi(2);
        let h_hat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let l_hat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            h_hat,
            l_hat,
            y: 0.,
            y1,
            rng,
            tf,
            lastdy: 0,
            lastdys: vec![0.; k],
            count: 0,
            last_set: LastSet::NotSet,
            ub: 2. * delta - gamma + y1,
            lb: 2. * delta - 2. * gamma + y1,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized, D: na::Dim + na::DimName + Sized + std::marker::Copy>
    StochasticSystem<State<D>> for Feedback2<'a, R, D>
where
    D: na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    fn system(&mut self, t: f64, dt: f64, rho: &State<D>, drho: &mut State<D>, dw: &Vec<f64>) {
        let avg = if t > self.tf {
            self.y / (self.count as f64 * dt)
            // self.count provides a better behaviour
            // (self.lastdys.len() as f64 * dt)
        } else {
            0.
        };

        let corr = if avg <= self.lb {
            0.
        } else if avg >= self.ub {
            1.
        } else if self.last_set == LastSet::NotSet || self.last_set == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.last_set = if avg <= self.lb {
            LastSet::LeGammaSet
        } else if avg >= self.ub {
            LastSet::GeGamma2Set
        } else {
            self.last_set
        };

        let h_hat = self.h_hat + self.f0.scale(corr);
        *drho = rouchonstep(dt, &rho, &h_hat, &self.l_hat, dw[0]);
        let dy = self.measurement(rho, dt, dw[0]);
        self.y += dy - self.lastdys[self.lastdy];
        self.lastdys[self.lastdy] = dy;
        self.lastdy = (self.lastdy + 1) % self.lastdys.len();
        if self.count < self.lastdys.len() {
            self.count += 1;
        }
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &State<D>, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
