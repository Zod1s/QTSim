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
    let a = nalgebra::Matrix3::new(-2., 3., 0., 0., -1., 0., 1., 0., -1.);
    let q = -nalgebra::Matrix3::identity();
    let p = lyapunov::lyapunovequation(&a, &q)?;
    println!("{a}");
    println!("{p}");

    Ok(())
}
