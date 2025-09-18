use crate::solver::StochasticSystem;
use crate::utils::*;
use crate::wiener;

#[derive(Debug)]
/// New feedback with both F0 and F1 for multilevel systems, using the ideal yt
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
    y1: f64,
    lb: f64,
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
            y1,
            rng,
            lb: 2. * delta - gamma + y1,
            wiener: wiener::Wiener::new(),
        }
    }
}

impl<'a, R: wiener::Rng + ?Sized, D: na::Dim + na::DimName + Sized + std::marker::Copy>
    StochasticSystem<State<D>> for Feedback<'a, R, D>
where
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    <na::DefaultAllocator as na::allocator::Allocator<D, D>>::Buffer<na::Complex<f64>>:
        std::marker::Copy,
{
    fn system(&mut self, t: f64, dt: f64, rho: &State<D>, dx: &mut State<D>, dw: &Vec<f64>) {
        let y = ((self.l + self.l.adjoint()) * rho).trace().re;
        let corr = if y > self.lb { 1. } else { 0. };

        let id = Operator::identity();
        let hhat = self.hhat + self.f0.scale(corr);
        let fst = (hhat * na::Complex::I + self.lhat.adjoint() * self.lhat.scale(0.5)).scale(dt);
        let snd = self
            .lhat
            .scale((self.lhat * rho + rho * self.lhat.adjoint()).trace().re * dt + dw[0]);
        let thd = (self.lhat * self.lhat).scale(dw[0].powi(2) - dt).scale(0.5);

        let m = id - fst + snd + thd;

        let num = m * rho * m.adjoint();
        *dx = num.scale(1. / num.trace().re) - rho;
    }

    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>) {
        *dw = self.wiener.sample_vector(dt, 1, self.rng);
    }

    fn measurement(&self, x: &State<D>, dt: f64, dw: f64) -> f64 {
        (self.l * x + x * self.l.adjoint()).trace().re * dt + dw
    }
}
