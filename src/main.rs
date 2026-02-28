#![allow(unused)]

mod dataplots;
mod examples;
mod plots;
mod solver;
mod systems;
mod utils;
mod wiener;

use crate::utils::*;
use rayon::prelude::*;
use std::thread;

// const NUMJOBS: usize = 1;
// const NUMSIMS: usize = 7;
// const NUMTHREADS: usize = NUMJOBS * NUMSIMS;

fn main() -> utils::SolverResult<()> {
    // examples::actualfeed::actualfeed()
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(NUMTHREADS.min(num_cpus::get()).max(1))
    //     .build_global()
    //     .expect("Could not create threadpool");

    examples::parallel::parallel_3d();

    // let thread1 = thread::spawn(|| {
    //     examples::parallel::parallel_3d();
    // });
    //
    // let thread2 = thread::spawn(|| {
    //     examples::parallel::parallel_anti_heis();
    // });
    //
    // thread1.join().unwrap();
    // thread2.join().unwrap();

    // rayon::scope(|s| {
    //     s.spawn(|s| {
    //         examples::parallel::parallel_3d(false);
    //     });
    //     s.spawn(|s| {
    //         examples::parallel::parallel_heis(false);
    //     });
    // });

    // dataplots::plot("./anti_heis.csv", "parallel_anti_heis", true);
    dataplots::plot("./3d3.csv", "parallel_3d3", true);

    // let vectors: Vec<na::Vector2<na::Complex<f64>>> = vec![na::Vector2::x(), na::Vector2::y()];
    // let id = na::Matrix2::<na::Complex<f64>>::identity();
    // let abtrace = (0..4)
    //     .map(|i| vectors[i / 2].kronecker(&vectors[i % 2]).kronecker(&id))
    //     .collect::<Vec<na::SMatrix<na::Complex<f64>, 8, 2>>>();
    // let bctrace = (0..4)
    //     .map(|i| id.kronecker(&vectors[i / 2]).kronecker(&vectors[i % 2]))
    //     .collect::<Vec<na::SMatrix<na::Complex<f64>, 8, 2>>>();
    // let actrace = (0..4)
    //     .map(|i| vectors[i / 2].kronecker(&id).kronecker(&vectors[i % 2]))
    //     .collect::<Vec<na::SMatrix<na::Complex<f64>, 8, 2>>>();
    //
    // let h = -ferromagnetictriangle(&[1., 1., 2.]);
    // let eigen = h.symmetric_eigen();
    // let col = eigen.eigenvectors.column(2);
    // println!("{}", col);
    // println!("2/1: {}", col[2].re / col[1].re);
    // println!("4/1: {}", col[4].re / col[1].re);
    // println!("4/2: {}", col[4].re / col[2].re);
    //
    // let eigens = vec![
    //     (vectors[0].kronecker(&vectors[1]).kronecker(&vectors[0])
    //         - vectors[0].kronecker(&vectors[0]).kronecker(&vectors[1]))
    //     .scale(1. / 2f64.sqrt()),
    //     (vectors[1].kronecker(&vectors[0]).kronecker(&vectors[1])
    //         - vectors[0].kronecker(&vectors[1]).kronecker(&vectors[1]))
    //     .scale(1. / 2f64.sqrt()),
    //     (vectors[1]
    //         .kronecker(&vectors[0])
    //         .kronecker(&vectors[0])
    //         .scale(2.)
    //         - vectors[0].kronecker(&vectors[1]).kronecker(&vectors[0])
    //         - vectors[0].kronecker(&vectors[0]).kronecker(&vectors[1]))
    //     .scale(1. / 6f64.sqrt()),
    //     (vectors[1]
    //         .kronecker(&vectors[1])
    //         .kronecker(&vectors[0])
    //         .scale(2.)
    //         - vectors[1].kronecker(&vectors[0]).kronecker(&vectors[1])
    //         - vectors[0].kronecker(&vectors[1]).kronecker(&vectors[1]))
    //     .scale(1. / 6f64.sqrt()),
    // ];
    // println!("{}", h * eigens[0]);

    // for vec in eigens {
    //     let rho = &vec * vec.adjoint();
    //     let pta = partialtracequbit(&rho, &bctrace);
    //     let ptb = partialtracequbit(&rho, &actrace);
    //     let ptc = partialtracequbit(&rho, &abtrace);
    //     let pura = (pta * pta).trace().re;
    //     let purb = (ptb * ptb).trace().re;
    //     let purc = (ptc * ptc).trace().re;
    //     println!(
    //         "Partial purities for {:.4}: TrA = {:.4}, TrB = {:.4}, TrC = {:.4}",
    //         vec, pura, purb, purc
    //     );
    // }

    // let state: na::SVector<na::Complex<f64>, 8> = na::Vector2::x()
    //     .kronecker(&na::Vector2::y())
    //     .kronecker(&na::Vector2::x());
    // let state = &state * &state.adjoint();
    // println!("{state}");
    // println!(
    //     "TrAB[state]: {}",
    //     abtrace
    //         .iter()
    //         .map(|ab| ab.adjoint() * state * ab)
    //         .sum::<Operator<na::U2>>()
    // );
    // println!(
    //     "TrBC[state]: {}",
    //     bctrace
    //         .iter()
    //         .map(|bc| bc.adjoint() * state * bc)
    //         .sum::<Operator<na::U2>>()
    // );
    // println!(
    //     "TrAC[state]: {}",
    //     actrace
    //         .iter()
    //         .map(|ac| ac.adjoint() * state * ac)
    //         .sum::<Operator<na::U2>>()
    // );

    utils::clean_up_python();

    Ok(())
}
