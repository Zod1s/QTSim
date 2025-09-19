#![allow(unused)]

mod examples;
mod plots;
mod solver;
mod systems;
mod utils;
mod wiener;

fn main() -> utils::SolverResult<()> {
    examples::multilevel::multilevel()
}
