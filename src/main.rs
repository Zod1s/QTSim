#![allow(unused)]

mod examples;
mod kron;
mod lyapunov;
mod plots;
mod solver;
mod systems;
mod utils;
mod wiener;

fn main() -> utils::SolverResult<()> {
    kron::lyapunovtrend()
}
