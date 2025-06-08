use crate::utils::*;
use crate::wiener::Wiener;
// use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use itertools::Itertools;
use nalgebra as na;
use rand::rngs::ThreadRng;
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
    fn system(&self, t: f64, x: &V, dx: &mut V, dw: &Vec<f64>);
    fn generate_noises(&self, step_size: f64, dw: &mut Vec<f64>);
    fn measurement(&self, x: &V, noise: f64) -> f64;
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
pub struct StochasticSolver<V, F>
where
    F: StochasticSystem<V>,
{
    f: F,
    t0: f64,
    t_end: f64,
    x0: V,
    step_size: f64,
    num_steps: usize,
    results: StochasticSolverOutput<V>,
}

impl<D: na::Dim, F> StochasticSolver<State<D>, F>
where
    F: StochasticSystem<State<D>>,
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
            num_steps,
            results: StochasticSolverOutput::with_capacity(num_steps),
        }
    }

    pub fn integrate(&mut self) -> SolverResult<()> {
        // TODO modify to handle multiple outputs
        let mut dw = Vec::new();
        self.f.generate_noises(self.step_size, &mut dw);
        let mut dy = self.f.measurement(&self.x0, dw[0]);

        self.results.push(self.t0, self.x0.clone(), dy);

        let mut t = self.t0;
        let mut x = self.x0.clone();
        let shape = self.x0.shape_generic();
        let mut dx = na::OMatrix::zeros_generic(shape.0, shape.1);

        for _ in 0..self.num_steps {
            self.f.system(t, &x, &mut dx, &dw);
            t += self.step_size;
            x += &dx;
            self.f.generate_noises(self.step_size, &mut dw);
            dy = self.f.measurement(&x, dw[0]);

            self.results.push(t, x.clone(), dy);
        }

        Ok(())
    }

    //     fn step(&self, state: &State<D>, rng: &mut ThreadRng) -> (State<D>, Vec<f64>) {
    //         let id = na::DMatrix::identity(self.size, self.size);
    //
    //         let fst = (&self.hi
    //             + self
    //                 .ls
    //                 .iter()
    //                 .map(|l| l.adjoint() * l.scale(0.5))
    //                 .sum::<Operator<D>>())
    //         .scale(self.dt);
    //
    //         let leta = self
    //             .ls
    //             .iter()
    //             .zip(&self.sqrtetas)
    //             .map(|(l, sqrteta)| l.scale(*sqrteta));
    //
    //         let w = self.wiener.sample_vector(self.dt, self.ls.len(), rng);
    //         let letaw = leta.zip(w.clone());
    //         let snd = letaw
    //             .clone()
    //             .map(|(l, w)| &l * ((&l * state + state * &l.adjoint()).trace() * self.dt + w))
    //             .sum::<Operator<D>>();
    //
    //         let letawr = letaw.zip(0..self.ls.len());
    //         let thd = letawr
    //             .clone()
    //             .cartesian_product(letawr)
    //             .map(|(((letar, wr), r), ((letas, ws), s))| {
    //                 letar * letas.scale(0.5 * (wr * ws - delta(&r, &s) * self.dt))
    //             })
    //             .sum::<Operator<D>>();
    //
    //         let mn = id - fst + snd + thd;
    //
    //         let num = &mn * state * &mn.adjoint()
    //             + self
    //                 .ls
    //                 .iter()
    //                 .zip(self.etas)
    //                 .map(|(l, eta)| l * state * l.adjoint().scale(self.dt * (1. - eta)))
    //                 .sum::<Operator<D>>();
    //
    //         (num.scale(1. / num.trace().re), self.measurement(state, w))
    //     }

    // fn measurement(&self, state: &State<D>, &wieners: Vec<f64>) -> Vec<f64> {
    //     self.ls
    //         .iter()
    //         .zip(&self.sqrtetas)
    //         .zip(wieners)
    //         .map(|((l, sqrteta), w)| {
    //             (l * state + state * l.adjoint()).trace().re * sqrteta * self.dt + w
    //         })
    //         .collect::<Vec<f64>>()
    // }

    //     pub fn trajectory(
    //         &self,
    //         final_time: f64,
    //     ) -> Result<(Vec<State<D>>, Vec<Vec<f64>>), SolverError> {
    //         if final_time <= 0. {
    //             return Err(SolverError::NegativeFinalTime);
    //         }
    //         let n_samples = (final_time / self.dt).floor() as usize;
    //         let mut measurements = Vec::with_capacity(n_samples);
    //         let mut states = Vec::with_capacity(n_samples);
    //         states.push(self.state.clone());
    //
    //         let mut rng = rand::rng();
    //
    //         for i in 1..n_samples {
    //             let (state, measurement) = self.step(&states[i - 1], &mut rng);
    //             states.push(state);
    //             measurements.push(measurement);
    //         }
    //
    //         // we add the measurement of the last state
    //         let noises = self.wiener.sample_vector(self.dt, self.ls.len(), &mut rng);
    //         measurements.push(self.measurement(&states[states.len() - 1], noises));
    //
    //         Ok((states, measurements))
    //     }

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

// pub struct StochasticSolver<'a, D>
// where
//     D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
//     na::DefaultAllocator: na::allocator::Allocator<D, D>,
// {
//     pub wiener: Wiener,
//     pub state: &'a State<D>,
//     pub h: &'a Operator<D>,
//     pub hi: Operator<D>,
//     pub ls: &'a Vec<Operator<D>>,
//     pub etas: &'a Vec<f64>,
//     pub sqrtetas: Vec<f64>,
//     pub size: usize,
//     pub dt: f64,
// }
//
// impl<'a, D> StochasticSolver<'a, D>
// where
//     D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
//     na::DefaultAllocator: na::allocator::Allocator<D, D>,
//     na::DefaultAllocator: na::allocator::Allocator<D>,
//     na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
// {
//     pub fn new(
//         init_state: &'a State<D>,
//         h: &'a Operator<D>,
//         ls: &'a Vec<Operator<D>>,
//         etas: &'a Vec<f64>,
//         dt: f64,
//     ) -> Result<Self, SolverError> {
//         let state_shape = init_state.shape();
//         let hi = h * na::Complex::I;
//
//         if dt <= 0. {
//             return Err(SolverError::NotPositiveDt(dt));
//         }
//
//         for eta in etas.iter() {
//             if *eta < 0. || *eta > 1. {
//                 return Err(SolverError::InvalidEfficiency(*eta));
//             }
//         }
//
//         if etas.len() != ls.len() {
//             return Err(SolverError::NoiseEfficiencyMismatch(etas.len(), ls.len()));
//         }
//
//         // check dimensions
//
//         check_hermiticity(h)?;
//         check_state(init_state)?;
//
//         Ok(Self {
//             wiener: Wiener::new(),
//             state: init_state,
//             h,
//             hi,
//             ls,
//             sqrtetas: etas.iter().map(|x| x.sqrt()).collect(),
//             etas,
//             size: state_shape.0,
//             dt,
//         })
//     }
//
//     fn step(&self, state: &State<D>, rng: &mut ThreadRng) -> (State<D>, Vec<f64>) {
//         let id = na::DMatrix::identity(self.size, self.size);
//
//         let fst = (&self.hi
//             + self
//                 .ls
//                 .iter()
//                 .map(|l| l.adjoint() * l.scale(0.5))
//                 .sum::<Operator<D>>())
//         .scale(self.dt);
//
//         let leta = self
//             .ls
//             .iter()
//             .zip(&self.sqrtetas)
//             .map(|(l, sqrteta)| l.scale(*sqrteta));
//
//         let w = self.wiener.sample_vector(self.dt, self.ls.len(), rng);
//         let letaw = leta.zip(w.clone());
//         let snd = letaw
//             .clone()
//             .map(|(l, w)| &l * ((&l * state + state * &l.adjoint()).trace() * self.dt + w))
//             .sum::<Operator<D>>();
//
//         let letawr = letaw.zip(0..self.ls.len());
//         let thd = letawr
//             .clone()
//             .cartesian_product(letawr)
//             .map(|(((letar, wr), r), ((letas, ws), s))| {
//                 letar * letas.scale(0.5 * (wr * ws - delta(&r, &s) * self.dt))
//             })
//             .sum::<Operator<D>>();
//
//         let mn = id - fst + snd + thd;
//
//         let num = &mn * state * &mn.adjoint()
//             + self
//                 .ls
//                 .iter()
//                 .zip(self.etas)
//                 .map(|(l, eta)| l * state * l.adjoint().scale(self.dt * (1. - eta)))
//                 .sum::<Operator<D>>();
//
//         (num.scale(1. / num.trace().re), self.measurement(state, w))
//     }
//
//     fn measurement(&self, state: &State<D>, wieners: Vec<f64>) -> Vec<f64> {
//         self.ls
//             .iter()
//             .zip(&self.sqrtetas)
//             .zip(wieners)
//             .map(|((l, sqrteta), w)| {
//                 (l * state + state * l.adjoint()).trace().re * sqrteta * self.dt + w
//             })
//             .collect::<Vec<f64>>()
//     }
//
//     pub fn trajectory(
//         &self,
//         final_time: f64,
//     ) -> Result<(Vec<State<D>>, Vec<Vec<f64>>), SolverError> {
//         if final_time <= 0. {
//             return Err(SolverError::NegativeFinalTime);
//         }
//         let n_samples = (final_time / self.dt).floor() as usize;
//         let mut measurements = Vec::with_capacity(n_samples);
//         let mut states = Vec::with_capacity(n_samples);
//         states.push(self.state.clone());
//
//         let mut rng = rand::rng();
//
//         for i in 1..n_samples {
//             let (state, measurement) = self.step(&states[i - 1], &mut rng);
//             states.push(state);
//             measurements.push(measurement);
//         }
//
//         // we add the measurement of the last state
//         let noises = self.wiener.sample_vector(self.dt, self.ls.len(), &mut rng);
//         measurements.push(self.measurement(&states[states.len() - 1], noises));
//
//         Ok((states, measurements))
//     }
// }
//
