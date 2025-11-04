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
    // let a = nalgebra::Matrix2::new(-1., 1., 0., -2.);
    // let q = -nalgebra::Matrix2::identity();
    let a = nalgebra::Matrix3::new(-1., 1., 0., 0., -2., 3., 0., 0., -1.);
    let mut q = -nalgebra::Matrix3::identity();
    let mut scale = 1.;
    let mut info = 0;
    unsafe {
        lapack::dtrsyl(
            'T' as u8,
            'N' as u8,
            &[1],
            3,
            3,
            a.as_slice(),
            3,
            a.as_slice(),
            3,
            q.as_mut_slice(),
            3,
            &mut [scale],
            &mut info,
        )
    };
    println!("{info}");
    println!("{scale}");
    println!("{a}");
    println!("{q}");
    Ok(())
}
