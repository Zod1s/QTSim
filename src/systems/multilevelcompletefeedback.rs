use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug)]
/// New feedback with both F0 and F1 for multilevel systems, using the actual yt
pub struct Feedback<
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
    hhat: Operator<D>,
    lhat: Operator<D>,
    y: f64,
    y1: f64,
    lb: f64,
    ub: f64,
    tf: f64,
    lastset: LastSet,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized, D: na::Dim + na::DimName + Sized + std::marker::Copy>
    Feedback<'a, R, D>
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
        rng: &'a mut R,
    ) -> Self {
        let normal = Normal::standard();
        let tf = (normal.cdf((alpha + 1.) / 2.) / epsilon).powi(2);
        let hhat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            hhat,
            lhat,
            y: 0.,
            y1,
            rng,
            tf,
            lastset: LastSet::NotSet,
            ub: 2. * delta - gamma + y1,
            lb: 2. * delta - 2. * gamma + y1,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized, D: na::Dim + na::DimName + Sized + std::marker::Copy>
    StochasticSystem<State<D>> for Feedback<'a, R, D>
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
        } else if self.lastset == LastSet::NotSet || self.lastset == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.lastset = if avg <= self.lb {
            LastSet::LeGammaSet
        } else if avg >= self.ub {
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
    hhat: Operator<D>,
    lhat: Operator<D>,
    y: f64,
    y1: f64,
    lb: f64,
    ub: f64,
    tf: f64,
    lastdy: usize,
    lastdys: Vec<f64>,
    count: usize,
    lastset: LastSet,
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
        let hhat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            hhat,
            lhat,
            y: 0.,
            y1,
            rng,
            tf,
            lastdy: 0,
            lastdys: vec![0.; k],
            count: 0,
            lastset: LastSet::NotSet,
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
        } else {
            0.
        };

        let corr = if avg <= self.lb {
            0.
        } else if avg >= self.ub {
            1.
        } else if self.lastset == LastSet::NotSet || self.lastset == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.lastset = if avg <= self.lb {
            LastSet::LeGammaSet
        } else if avg >= self.ub {
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
