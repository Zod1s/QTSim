#![allow(unused)]

mod examples;
mod kron;
mod plots;
mod solver;
mod systems;
mod utils;
mod wiener;

fn main() -> utils::SolverResult<()> {
    kron::vectorisationexample();
    Ok(())
    // examples::output::output()
}
