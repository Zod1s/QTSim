use core::f64;

use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;
use statrs::distribution::{ContinuousCDF, Normal};

/// New feedback with both F0 and F1 for multilevel systems, using the actual yt
pub struct Controller<'a, R: wiener::Rng + ?Sized, const D: usize>
where
    na::Const<D>: na::Dim + na::DimName + Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<na::Const<D>, na::Const<D>>,
    <na::DefaultAllocator as na::allocator::Allocator<na::Const<D>, na::Const<D>>>::Buffer<
        na::Complex<f64>,
    >: std::marker::Copy,
{
    h: Operator<na::Const<D>>,
    l: Operator<na::Const<D>>,
    f0: Operator<na::Const<D>>,
    h_hat: Operator<na::Const<D>>,
    l_hat: Operator<na::Const<D>>,
    y: f64,
    y1: f64,
    lb: f64,
    ub: f64,
    tf: f64,
    count: usize,
    index: usize,
    last_dys: Vec<f64>,
    is_active: bool,
    max_active_time: f64,
    last_active_time: f64,
    last_set: LastSet,
    rng: &'a mut R,
    wiener: wiener::Wiener,
}

impl<'a, R: wiener::Rng + ?Sized, const D: usize> Controller<'a, R, D>
where
    na::Const<D>: na::Dim + na::DimName + Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<na::Const<D>, na::Const<D>>,
    <na::DefaultAllocator as na::allocator::Allocator<na::Const<D>, na::Const<D>>>::Buffer<
        na::Complex<f64>,
    >: std::marker::Copy,
{
    pub fn new(
        h: Operator<na::Const<D>>,
        l: Operator<na::Const<D>>,
        hc: Operator<na::Const<D>>,
        f0: Operator<na::Const<D>>,
        f1: Operator<na::Const<D>>,
        y1: f64,
        k: usize,
        delta: f64,
        gamma: f64,
        alpha: f64,
        epsilon: f64,
        theta: f64,
        b: f64,
        lmin: f64,
        lmax: f64,
        rng: &'a mut R,
    ) -> Self {
        let normal = Normal::standard();
        let tf = (normal.cdf((alpha + 1.) / 2.) / epsilon).powi(2);
        let h_hat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let l_hat = l - f1 * na::Complex::I;
        let max_active_time =
            lmax * (lmax * b.powi(2) * (D as f64) / (lmin * theta.powi(2) * (D as f64 - 1.))).ln();

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
            index: 0,
            last_dys: vec![0.; k],
            count: 0,
            is_active: false,
            max_active_time,
            last_active_time: -f64::INFINITY,
            last_set: LastSet::NotSet,
            ub: 2. * delta - gamma + y1,
            lb: 2. * delta - 2. * gamma + y1,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized, const D: usize> StochasticSystem<State<na::Const<D>>>
    for Controller<'a, R, D>
where
    na::Const<D>: na::DimSub<na::Const<1>> + na::Dim + na::DimName + Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<na::Const<D>, na::Const<D>>,
    <na::DefaultAllocator as na::allocator::Allocator<na::Const<D>, na::Const<D>>>::Buffer<
        na::Complex<f64>,
    >: std::marker::Copy,
{
    fn system(
        &mut self,
        t: f64,
        dt: f64,
        rho: &State<na::Const<D>>,
        drho: &mut State<na::Const<D>>,
        dw: &Vec<f64>,
    ) {
        let corr = if !self.is_active
        // && t - self.last_active_time < self.last_dys.len() as f64 * dt
        {
            // If the control is inactive and not enough time has passed since we have turned it
            // off, the control is still set to zero independently of the measurements
            // 0.
            // } else if !self.is_active {
            let avg = if t > self.tf {
                self.y / (self.count as f64 * dt)
            } else {
                0.
            };

            let corr = if avg <= self.lb {
                0.
            } else if avg >= self.ub {
                1.
            } else if self.last_set != LastSet::GeGamma2Set {
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

            if corr == 1. {
                self.last_active_time = t;
                self.is_active = true;
            }

            corr
        } else if t - self.last_active_time < self.max_active_time {
            let avg = if t > self.tf {
                self.y / (self.count as f64 * dt)
            } else {
                0.
            };

            let corr = if avg <= self.lb {
                0.
            } else if avg >= self.ub {
                1.
            } else if self.last_set != LastSet::GeGamma2Set {
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

            if corr == 0. {
                self.last_active_time = t;
                self.is_active = false;
            }

            corr
        } else {
            self.last_active_time = t;
            self.is_active = false;
            self.last_set = LastSet::NotSet;
            0.
        };

        let h_hat = self.h_hat + self.f0.scale(corr);
        *drho = rouchonstep(dt, &rho, &h_hat, &self.l_hat, dw[0]);
        let dy = self.measurement(rho, dt, dw[0]);
        self.y += dy - self.last_dys[self.index];
        self.last_dys[self.index] = dy;
        self.index = (self.index + 1) % self.last_dys.len();
        if self.count < self.last_dys.len() {
            self.count += 1;
        }
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &State<na::Const<D>>, dt: f64, dw: f64) -> f64 {
        ((self.l + self.l.adjoint()) * x).trace().re * dt + dw
    }
}
