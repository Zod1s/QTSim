use crate::utils::*;
use crate::wiener::Wiener;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use itertools::Itertools;
use nalgebra as na;

pub struct Solver<'a> {
    wiener: Wiener,
    state: &'a State,
    h: &'a Operator,
    hi: Operator,
    ls: &'a Vec<Operator>,
    etas: &'a Vec<f64>,
    sqrtetas: Vec<f64>,
    size: usize,
    dt: f64,
}

impl<'a> Solver<'a> {
    pub fn new(
        init_state: &'a State,
        h: &'a Operator,
        ls: &'a Vec<Operator>,
        etas: &'a Vec<f64>,
        dt: f64,
    ) -> Result<Self, Error> {
        let state_shape = init_state.shape();
        let h_shape = h.shape();
        let hi = h * na::Complex::I;

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

    fn step(&mut self, state: &State) -> State {
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

        let w = self.wiener.sample_vector(self.dt, self.ls.len());
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

        num.scale(1. / num.trace().re)

        // let y = self
        //     .ls
        //     .iter()
        //     .zip(&self.sqrtetas)
        //     .zip(w)
        //     .map(|((l, sqrteta), w)| {
        //         (l * &new_state + &new_state * l.adjoint()).trace().re * sqrteta * self.dt + w
        //     })
        //     .collect::<Vec<f64>>();
        //
        // (new_state, y)
    }

    pub fn trajectory(&mut self, final_time: f64) -> Result<Vec<State>, Error> {
        if final_time <= 0. {
            return Err(Error::NegativeFinalTime);
        }
        let n_samples = (final_time / self.dt).floor() as usize;
        let mut states = Vec::with_capacity(n_samples);
        states.push(self.state.clone());

        // let bar = ProgressBar::new((n_samples - 1) as u64).with_style(
        //     ProgressStyle::default_bar()
        //         .template("Sample: [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:}")
        //         .unwrap(),
        // );

        for i in 1..n_samples {
            // bar.inc(1);
            states.push(self.step(&states[i - 1]))
        }

        // bar.finish();

        Ok(states)
    }
}
