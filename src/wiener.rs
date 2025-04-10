pub(crate) use rand::prelude::*;
use rand_distr;

pub struct Wiener {
    distr: rand_distr::Normal<f64>,
}

impl Wiener {
    pub fn new() -> Self {
        Self {
            distr: rand_distr::Normal::new(0.0, 1.0)
                .expect("Could not create a normal distribution"),
        }
    }

    pub fn sample_increment(&self, dt: f64, rng: &mut ThreadRng) -> f64 {
        self.distr.sample(rng) * dt.sqrt()
    }

    pub fn sample_vector(&self, dt: f64, n: usize, rng: &mut ThreadRng) -> Vec<f64> {
        (0..n).map(|_| self.sample_increment(dt, rng)).collect()
    }
}
