use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;
use rand::{rngs::StdRng, SeedableRng};

#[derive(Debug)]
/// New feedback with both F0 and F1 for multilevel systems, using the ideal yt
pub struct Feedback<D: na::Dim + na::DimName + Sized + std::marker::Copy>
where
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    h: Operator<D>,
    l: Operator<D>,
    f0: Operator<D>,
    hhat: Operator<D>,
    lhat: Operator<D>,
    y1: f64,
    lb: f64,
    ub: f64,
    lastset: LastSet,
    rng: StdRng,
    wiener: wiener::Wiener,
    eta: f64,
}

impl<D: na::Dim + na::DimName + Sized + std::marker::Copy> Feedback<D>
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
        eta: f64,
        y1: f64,
        delta: f64,
        gamma: f64,
        rng: Option<u64>,
    ) -> Self {
        let hhat = h + hc + (f1 * l + l.adjoint() * f1).scale(0.5);
        let lhat = l - f1 * na::Complex::I;

        Self {
            h,
            l,
            f0,
            hhat,
            lhat,
            y1,
            rng: StdRng::seed_from_u64(rng.unwrap_or(0)),
            lastset: LastSet::NotSet,
            ub: 2. * delta - gamma + y1,
            lb: 2. * delta - 2. * gamma + y1,
            wiener: wiener::Wiener::new(),
            eta,
        }
    }
}

impl<D: na::Dim + na::DimName + Sized + std::marker::Copy> StochasticSystem<State<D>>
    for Feedback<D>
where
    D: na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    fn system(&mut self, t: f64, dt: f64, rho: &State<D>, drho: &mut State<D>, dw: &Vec<f64>) {
        let y = ((self.l + self.l.adjoint()) * rho).trace().re;

        let corr = if y <= self.lb {
            0.
        } else if y >= self.ub {
            1.
        } else if self.lastset == LastSet::NotSet || self.lastset == LastSet::LeGammaSet {
            0.
        } else {
            1.
        };

        self.lastset = if y <= self.lb {
            LastSet::LeGammaSet
        } else if y >= self.ub {
            LastSet::GeGamma2Set
        } else {
            self.lastset
        };

        let hhat = self.hhat + self.f0.scale(corr);
        *drho = rouchonstep(dt, &rho, &hhat, &self.lhat, dw[0], self.eta);
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, &mut self.rng);
    }

    fn measurement(&self, x: &State<D>, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
