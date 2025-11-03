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
    let a = nalgebra::Matrix2::new(-1., 1., 0., -2.);
    let q = -nalgebra::Matrix2::identity();
    let p = lyapunov::lyapunov2x2(&a, &q)?;
    println!("{p}");
    Ok(())
}
