use rand::prelude::*;
use rand_distr;

pub struct Wiener {
    distr: rand_distr::Normal<f64>,
    rng: ThreadRng,
}

impl Wiener {
    pub fn new() -> Self {
        Self {
            distr: rand_distr::Normal::new(0.0, 1.0)
                .expect("Could not create a normal distribution"),
            rng: rand::rng(),
        }
    }

    pub fn sample_increment(&mut self, dt: f64) -> f64 {
        self.distr.sample(&mut self.rng) * dt.sqrt()
    }

    pub fn sample_vector(&mut self, dt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample_increment(dt)).collect()
    }
}
