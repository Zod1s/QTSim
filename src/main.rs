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
use std::process::Command;
use std::thread;

const NUMJOBS: usize = 1;
const NUMSIMS: usize = 7;
const NUMTHREADS: usize = NUMJOBS * NUMSIMS;

fn main() -> utils::SolverResult<()> {}
