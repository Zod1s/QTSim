use crate::utils::*;
use nalgebra as na;
use rand_distr::num_traits::ToPrimitive;

pub trait System<V> {
    fn system(&self, t: f64, x: &V, dx: &mut V);
}

#[derive(Debug, Clone)]
pub struct SolverOutput<V>(Vec<f64>, Vec<V>);

impl<V> SolverOutput<V> {
    pub fn new(x: Vec<f64>, y: Vec<V>) -> Self {
        Self(x, y)
    }

    pub fn with_capacity(n: usize) -> Self {
        Self(Vec::with_capacity(n), Vec::with_capacity(n))
    }

    pub fn push(&mut self, x: f64, y: V) {
        self.0.push(x);
        self.1.push(y);
    }

    pub fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
        self.1.append(&mut other.1);
    }

    /// Returns a pair that contains references to the internal vectors
    pub fn get(&self) -> (&Vec<f64>, &Vec<V>) {
        (&self.0, &self.1)
    }
}

impl<V> Default for SolverOutput<V> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

pub struct Rk4<V, F>
where
    F: System<V>,
{
    f: F,
    t0: f64,
    t_end: f64,
    x0: V,
    step_size: f64,
    half_step: f64,
    num_steps: usize,
    results: SolverOutput<V>,
}

impl<D: na::Dim, F> Rk4<State<D>, F>
where
    F: System<State<D>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    pub fn new(f: F, t0: f64, x0: State<D>, t_end: f64, step_size: f64) -> Self {
        let num_steps = (((t_end - t0) / step_size).ceil()).to_usize().unwrap();

        Self {
            f,
            t0,
            t_end,
            x0,
            step_size,
            half_step: step_size / 2.,
            num_steps,
            results: SolverOutput::with_capacity(num_steps),
        }
    }

    pub fn integrate(&mut self) -> SolverResult<()> {
        self.results.push(self.t0, self.x0.clone());
        let mut t = self.t0;
        let mut x = self.x0.clone();
        let shape = self.x0.shape_generic();
        let mut k1 = na::OMatrix::zeros_generic(shape.0, shape.1);
        let mut k2 = na::OMatrix::zeros_generic(shape.0, shape.1);
        let mut k3 = na::OMatrix::zeros_generic(shape.0, shape.1);
        let mut k4 = na::OMatrix::zeros_generic(shape.0, shape.1);

        for _ in 0..self.num_steps {
            self.f.system(t, &x, &mut k1);
            self.f.system(
                t + self.half_step,
                &(&x + &k1.scale(self.half_step)),
                &mut k2,
            );
            self.f.system(
                t + self.half_step,
                &(&x + &k2.scale(self.half_step)),
                &mut k3,
            );
            self.f.system(
                t + self.step_size,
                &(&x + &k3.scale(self.step_size)),
                &mut k4,
            );

            t += self.step_size;
            x += (&k1 + &k2.scale(2.) + &k3.scale(2.) + &k4).scale(self.step_size / 6.);
            self.results.push(t, x.clone());
        }

        Ok(())
    }

    /// Getter for the independent variable's output.
    pub fn t_out(&self) -> &Vec<f64> {
        self.results.get().0
    }

    /// Getter for the dependent variables' output.
    pub fn x_out(&self) -> &Vec<State<D>> {
        self.results.get().1
    }

    /// Getter for the results type, a pair of independent and dependent variables
    pub fn results(&self) -> &SolverOutput<State<D>> {
        &self.results
    }

    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    pub fn step_size(&self) -> f64 {
        self.step_size
    }
}

pub trait StochasticSystem<V> {
    // need to have a mutable reference to update the total output signal y
    fn system(&mut self, t: f64, dt: f64, x: &V, dx: &mut V, dw: &Vec<f64>);
    fn generate_noises(&mut self, dt: f64, dw: &mut Vec<f64>);
    // handles only single measurement for now
    fn measurement(&self, x: &V, dt: f64, dw: f64) -> f64;
}

#[derive(Debug, Clone)]
pub struct StochasticSolverOutput<V>(Vec<f64>, Vec<V>, Vec<f64>);

impl<V> StochasticSolverOutput<V> {
    pub fn new(x: Vec<f64>, y: Vec<V>, m: Vec<f64>) -> Self {
        Self(x, y, m)
    }

    pub fn with_capacity(n: usize) -> Self {
        Self(
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        )
    }

    pub fn push(&mut self, x: f64, y: V, m: f64) {
        self.0.push(x);
        self.1.push(y);
        self.2.push(m);
    }

    pub fn append(&mut self, mut other: Self) {
        self.0.append(&mut other.0);
        self.1.append(&mut other.1);
        self.2.append(&mut other.2);
    }

    /// Returns a pair that contains references to the internal vectors
    pub fn get(&self) -> (&Vec<f64>, &Vec<V>, &Vec<f64>) {
        (&self.0, &self.1, &self.2)
    }
}

impl<V> Default for StochasticSolverOutput<V> {
    fn default() -> Self {
        Self(Default::default(), Default::default(), Default::default())
    }
}
pub struct StochasticSolver<'a, V, F>
where
    F: StochasticSystem<V>,
{
    f: &'a mut F,
    t0: f64,
    t_end: f64,
    x0: V,
    step_size: f64,
    num_steps: usize,
    results: StochasticSolverOutput<V>,
}

impl<'a, D: na::Dim, F> StochasticSolver<'a, State<D>, F>
where
    F: StochasticSystem<State<D>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    pub fn new(f: &'a mut F, t0: f64, x0: State<D>, t_end: f64, step_size: f64) -> Self {
        let num_steps = (((t_end - t0) / step_size).ceil()).to_usize().unwrap();

        Self {
            f,
            t0,
            t_end,
            x0,
            step_size,
            num_steps,
            results: StochasticSolverOutput::with_capacity(num_steps),
        }
    }

    pub fn integrate(&mut self) -> SolverResult<()> {
        // TODO modify to handle multiple outputs
        let mut dw = Vec::new();
        self.f.generate_noises(self.step_size, &mut dw);
        let mut dy = self.f.measurement(&self.x0, self.step_size, dw[0]);

        self.results.push(self.t0, self.x0.clone(), dy);

        let mut t = self.t0;
        let mut x = self.x0.clone();
        let shape = self.x0.shape_generic();
        let mut dx = na::OMatrix::zeros_generic(shape.0, shape.1);

        for _ in 0..self.num_steps {
            self.f.system(t, self.step_size, &x, &mut dx, &dw);
            t += self.step_size;
            x += &dx;
            self.f.generate_noises(self.step_size, &mut dw);
            dy = self.f.measurement(&x, self.step_size, dw[0]);

            self.results.push(t, x.clone(), dy);
        }

        Ok(())
    }

    /// Getter for the independent variable's output.
    pub fn t_out(&self) -> &Vec<f64> {
        self.results.get().0
    }

    /// Getter for the dependent variables' output.
    pub fn x_out(&self) -> &Vec<State<D>> {
        self.results.get().1
    }

    /// Getter for the measurements.
    pub fn y_out(&self) -> &Vec<f64> {
        self.results.get().2
    }

    /// Getter for the results type, a pair of independent and dependent variables
    pub fn results(&self) -> &StochasticSolverOutput<State<D>> {
        &self.results
    }

    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    pub fn step_size(&self) -> f64 {
        self.step_size
    }
}
