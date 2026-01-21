use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Debug)]
pub struct SSE<D>
where
    D: na::Dim + na::DimName + Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    h: Operator<D>,
    l: Operator<D>,
    rng: StdRng,
    wiener: wiener::Wiener,
}

impl<D> SSE<D>
where
    D: na::Dim + na::DimName + Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    pub fn new(h: Operator<D>, l: Operator<D>, seed: Option<u64>) -> Self {
        Self {
            h,
            l,
            rng: StdRng::seed_from_u64(seed.unwrap_or(0)),
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<D> StochasticSystem<State<D>> for SSE<D>
where
    D: na::Dim + na::DimName + Sized + std::marker::Copy,
    D: na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    fn system(&mut self, _: f64, dt: f64, rho: &State<D>, drho: &mut State<D>, dw: &Vec<f64>) {
        *drho = rouchonstep(dt, &rho, &self.h, &self.l, dw[0]);
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, &mut self.rng);
    }

    fn measurement(&self, x: &State<D>, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
