use crate::utils::*;
use crate::wiener::Wiener;
use itertools::Itertools;
use nalgebra as na;

pub struct Solver<const T: usize> {
    wiener: Wiener,
    state: State<T>,
    h: Operator<T>,
    hi: Operator<T>,
    ls: Vec<Operator<T>>,
    etas: Vec<f64>,
    sqrtetas: Vec<f64>,
    size: usize,
    dt: f64,
}

impl<const T: usize> Solver<T> {
    pub fn new(
        init_state: State<T>,
        h: Operator<T>,
        ls: Vec<Operator<T>>,
        etas: Vec<f64>,
        dt: f64,
    ) -> Result<Self, Error> {
        let hi = &h * na::Complex::I;

        if dt <= 0. {
            return Err(Error::NotPositiveDt(dt));
        }

        for eta in etas.iter() {
            if *eta < 0. || *eta > 1. {
                return Err(Error::InvalidEfficiency(*eta));
            }
        }

        if etas.len() != ls.len() {
            return Err(Error::NoiseEfficiencyMismatch(etas.len(), ls.len()));
        }

        check_hermiticity(&h)?;
        check_state(&init_state)?;

        Ok(Self {
            wiener: Wiener::new(),
            state: init_state,
            h,
            hi,
            ls,
            sqrtetas: etas.iter().map(|x| x.sqrt()).collect(),
            etas,
            size: T,
            dt,
        })
    }

    fn step(&mut self, state: &State<T>) -> State<T> {
        let id = na::DMatrix::identity(self.size, self.size);
        let fst = (&self.hi
            + self
                .ls
                .iter()
                .map(|l| l.adjoint() * l.scale(0.5))
                .sum::<Operator<T>>())
        .scale(self.dt);
        let leta = self
            .ls
            .iter()
            .zip(&self.sqrtetas)
            .map(|(l, sqrteta)| l.scale(*sqrteta));
        let w = self.wiener.sample_vector(self.dt, self.ls.len());
        let letaw = leta.zip(w);
        let snd = letaw
            .clone()
            .map(|(l, w)| &l * ((&l * state + state * &l.adjoint()).trace() * self.dt + w))
            .sum::<Operator<T>>();
        let letawr = letaw.zip(0..self.ls.len());
        let thd = letawr
            .clone()
            .cartesian_product(letawr)
            .map(|(((letar, wr), r), ((letas, ws), s))| {
                letar * letas.scale(0.5 * (wr * ws - delta(&r, &s) * self.dt))
            })
            .sum::<Operator<T>>();
        let mn = id - fst + snd + thd;

        let num = &mn * state * &mn.adjoint()
            + self
                .ls
                .iter()
                .map(|l| l * state * l.adjoint().scale(self.dt))
                .sum::<Operator<T>>();
        num.scale(1. / num.trace().re)
    }

    pub fn trajectory(&mut self, final_time: f64) -> Result<Vec<State<T>>, Error> {
        if final_time <= 0. {
            return Err(Error::NegativeFinalTime);
        }
        let n_samples = (final_time / self.dt).floor() as usize;
        let mut states = Vec::with_capacity(n_samples);
        states.push(self.state.clone());

        for i in 1..n_samples {
            states.push(self.step(&states[i - 1]))
        }

        Ok(states)
    }
}
