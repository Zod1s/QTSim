use crate::utils::*;
use crate::wiener::Wiener;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use itertools::Itertools;
use nalgebra as na;
use rand::rngs::ThreadRng;
use rayon::prelude::*;
use std::{
    sync::{Arc, Mutex},
    thread::Thread,
};

pub struct Solver<'a> {
    pub wiener: Wiener,
    pub state: &'a State,
    pub h: &'a Operator,
    pub hi: Operator,
    pub ls: &'a Vec<Operator>,
    pub etas: &'a Vec<f64>,
    pub sqrtetas: Vec<f64>,
    pub size: usize,
    pub dt: f64,
}

impl<'a> Solver<'a> {
    pub fn new(
        init_state: &'a State,
        h: &'a Operator,
        ls: &'a Vec<Operator>,
        etas: &'a Vec<f64>,
        dt: f64,
    ) -> Result<Self, SolverError> {
        let state_shape = init_state.shape();
        let h_shape = h.shape();
        let hi = h * na::Complex::I;

        if dt <= 0. {
            return Err(SolverError::NotPositiveDt(dt));
        }

        for eta in etas.iter() {
            if *eta < 0. || *eta > 1. {
                return Err(SolverError::InvalidEfficiency(*eta));
            }
        }

        if etas.len() != ls.len() {
            return Err(SolverError::NoiseEfficiencyMismatch(etas.len(), ls.len()));
        }

        // check dimensions

        check_hermiticity(h)?;
        check_state(init_state)?;

        Ok(Self {
            wiener: Wiener::new(),
            state: init_state,
            h,
            hi,
            ls,
            sqrtetas: etas.iter().map(|x| x.sqrt()).collect(),
            etas,
            size: state_shape.0,
            dt,
        })
    }

    fn step(&self, state: &State, rng: &mut ThreadRng) -> (State, Vec<f64>) {
        let id = na::DMatrix::identity(self.size, self.size);

        let fst = (&self.hi
            + self
                .ls
                .iter()
                .map(|l| l.adjoint() * l.scale(0.5))
                .sum::<Operator>())
        .scale(self.dt);

        let leta = self
            .ls
            .iter()
            .zip(&self.sqrtetas)
            .map(|(l, sqrteta)| l.scale(*sqrteta));

        let w = self.wiener.sample_vector(self.dt, self.ls.len(), rng);
        let letaw = leta.zip(w.clone());
        let snd = letaw
            .clone()
            .map(|(l, w)| &l * ((&l * state + state * &l.adjoint()).trace() * self.dt + w))
            .sum::<Operator>();

        let letawr = letaw.zip(0..self.ls.len());
        let thd = letawr
            .clone()
            .cartesian_product(letawr)
            .map(|(((letar, wr), r), ((letas, ws), s))| {
                letar * letas.scale(0.5 * (wr * ws - delta(&r, &s) * self.dt))
            })
            .sum::<Operator>();

        let mn = id - fst + snd + thd;

        let num = &mn * state * &mn.adjoint()
            + self
                .ls
                .iter()
                .zip(self.etas)
                .map(|(l, eta)| l * state * l.adjoint().scale(self.dt * (1. - eta)))
                .sum::<Operator>();

        (num.scale(1. / num.trace().re), self.measurement(state, w))
    }

    fn measurement(&self, state: &State, wieners: Vec<f64>) -> Vec<f64> {
        self.ls
            .iter()
            .zip(&self.sqrtetas)
            .zip(wieners)
            .map(|((l, sqrteta), w)| {
                (l * state + state * l.adjoint()).trace().re * sqrteta * self.dt + w
            })
            .collect::<Vec<f64>>()
    }

    pub fn trajectory(&self, final_time: f64) -> Result<(Vec<State>, Vec<Vec<f64>>), SolverError> {
        if final_time <= 0. {
            return Err(SolverError::NegativeFinalTime);
        }
        let n_samples = (final_time / self.dt).floor() as usize;
        let mut measurements = Vec::with_capacity(n_samples);
        let mut states = Vec::with_capacity(n_samples);
        states.push(self.state.clone());

        let mut rng = rand::rng();

        for i in 1..n_samples {
            let (state, measurement) = self.step(&states[i - 1], &mut rng);
            states.push(state);
            measurements.push(measurement);
        }

        // we add the measurement of the last state
        let noises = self.wiener.sample_vector(self.dt, self.ls.len(), &mut rng);
        measurements.push(self.measurement(&states[states.len() - 1], noises));

        Ok((states, measurements))
    }

    // pub fn parallel_trajectories(
    //     &self,
    //     final_time: f64,
    //     par_instances: usize,
    // ) -> Result<(Vec<State>, Vec<Vec<f64>>), SolverError> {
    //     if final_time <= 0. {
    //         return Err(SolverError::NegativeFinalTime);
    //     }
    //
    //     let n_samples = (final_time / self.dt).floor() as usize;
    //
    //     panic!("")
    // }
}
