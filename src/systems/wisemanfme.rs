use crate::solver::{StochasticSystem, System};
use crate::utils::*;
use crate::wiener;

#[derive(Clone, Copy, Debug)]
pub struct WisemanFME<D: na::Dim>
where
    D: Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    hhat: Operator<D>,
    lhat: Operator<D>,
}

impl<D: na::Dim> WisemanFME<D>
where
    D: Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    pub fn new(h: Operator<D>, l: Operator<D>, f: Operator<D>) -> Self {
        let lhat = l - f * na::Complex::I;
        let hhat = h + (f * l + l.adjoint() * f).scale(0.5);
        Self { hhat, lhat }
    }
}

impl<D: na::Dim> System<State<D>> for WisemanFME<D>
where
    D: Sized + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    fn system(&self, _: f64, rho: &State<D>, drho: &mut State<D>) {
        *drho = -commutator(&self.hhat, rho) * na::Complex::I
            + self.lhat * rho * self.lhat.adjoint()
            - anticommutator(&(self.lhat.adjoint() * self.lhat), rho).scale(0.5);
    }
}
