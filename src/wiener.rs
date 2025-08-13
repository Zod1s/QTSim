pub(crate) use rand::prelude::*;

#[derive(Clone, Copy, Debug)]
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

    pub fn sample_increment<R: Rng + ?Sized>(&self, dt: f64, rng: &mut R) -> f64 {
        self.distr.sample(rng) * dt.sqrt()
    }

    pub fn sample_vector<R: Rng + ?Sized>(&self, dt: f64, n: usize, rng: &mut R) -> Vec<f64> {
        (0..n).map(|_| self.sample_increment(dt, rng)).collect()
    }
}
