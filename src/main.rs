#![allow(unused)]

mod dataplots;
mod examples;
// mod kron;
// mod lyapunov;
mod plots;
mod solver;
mod systems;
mod utils;
mod wiener;

use crate::utils::*;
use rayon::prelude::*;
use std::thread;

const NUMJOBS: usize = 2;
const NUMSIMS: usize = 7;
const NUMTHREADS: usize = NUMJOBS * NUMSIMS;

// Consider optimising \beta and \varepsilon for the original controller and compute the optimal
// window to reduce the variance under a certain threshold
fn main() -> utils::SolverResult<()> {
    // examples::actualfeed::actualfeed()
    rayon::ThreadPoolBuilder::new()
        .num_threads(NUMTHREADS.min(num_cpus::get()).max(1))
        .build_global()
        .expect("Could not create threadpool");

    // examples::parallel::parallel_3d();

    let thread1 = thread::spawn(|| {
        examples::parallel::parallel_3d();
    });

    let thread2 = thread::spawn(|| {
        examples::parallel::parallel_anti_heis();
    });

    thread1.join().unwrap();
    thread2.join().unwrap();

    // rayon::scope(|s| {
    //     s.spawn(|s| {
    //         examples::parallel::parallel_3d(false);
    //     });
    //     s.spawn(|s| {
    //         examples::parallel::parallel_heis(false);
    //     });
    // });

    // dataplots::plot("./heis.csv")
    Ok(())
}
