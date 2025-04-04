use lazy_static::lazy_static;
pub use nalgebra as na;
use nalgebra::ToTypenum;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

type TMul<T> = <T as Mul>::Output;

pub type State = na::DMatrix<na::Complex<f64>>;
pub type Operator = na::DMatrix<na::Complex<f64>>;

lazy_static! {
    pub static ref PAULI_X: na::DMatrix<na::Complex<f64>> = na::dmatrix![
        na::Complex::ZERO,
        na::Complex::ONE;
        na::Complex::ONE,
        na::Complex::ZERO
    ];
    pub static ref PAULI_Y: na::DMatrix<na::Complex<f64>> = na::dmatrix![
        na::Complex::ZERO,
        na::Complex::new(0., -1.);
        na::Complex::ONE,
        na::Complex::I
    ];
    pub static ref PAULI_Z: na::DMatrix<na::Complex<f64>> = na::dmatrix![
        na::Complex::ONE,
        na::Complex::ZERO;
        na::Complex::ZERO,
        na::Complex::new(-1., 0.)
    ];
}

pub fn commutator<'c, T>(a: &'c T, b: &'c T) -> <TMul<&'c T> as Sub>::Output
where
    &'c T: Mul,
    TMul<&'c T>: Sub,
{
    a * b - b * a
}

pub fn anticommutator<'c, T>(a: &'c T, b: &'c T) -> <TMul<&'c T> as Add>::Output
where
    &'c T: Mul,
    TMul<&'c T>: Add,
{
    a * b + b * a
}
pub fn delta<A: Eq>(x: &A, y: &A) -> f64 {
    if x == y {
        1.
    } else {
        0.
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Error {
    ShapeError((usize, usize), (usize, usize)),
    NotSquareError(usize, usize),
    NotPositiveDt(f64),
    InvalidEfficiency(f64),
    NotHermitian,
    NotPositiveSemidefinite,
    NotUnitaryTrace(f64),
    NegativeFinalTime,
    IncompatibleNoiseShapes,
    NotSquareNoises,
    NoiseEfficiencyMismatch(usize, usize),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ShapeError(shape1, shape2) => {
                write!(
                    f,
                    "Error, shapes {:?} and {:?} do not match",
                    shape1, shape2
                )
            }
            Error::NotSquareError(rows, cols) => write!(
                f,
                "Error, the given matrix is not square, with {} rows and {} cols",
                rows, cols
            ),
            Error::NotPositiveDt(dt) => write!(f, "Error, the given time step {} is not positive", dt),
            Error::InvalidEfficiency(eta) => write!(f, "Error, the given measurement efficiency {} is not valid since it is not in the interval [0, 1]",eta),
            Error::NotHermitian => write!(f, "Error, the given matrix is not hermitian"),
            Error::NotPositiveSemidefinite => write!(f, "Error, the given matrix is not positive semidefinite"),
            Error::NotUnitaryTrace(trace) => write!(f, "Error, the given matrix has trace {}, which is different from 1", trace),
            Error::NegativeFinalTime => write!(f, "Error, the final time of the simulation cannot be negative"),
            Error::IncompatibleNoiseShapes => write!(f, "Error, the noise operators have incompatible noise shapes"),
            Error::NotSquareNoises => write!(f, "Error, some of the noise operators are not square"),
            Error::NoiseEfficiencyMismatch(etas, ls) => write!(f, "Error, the number of etas, {}, is different from the number of ls, {}", etas, ls),
        }
    }
}

pub fn check_hermiticity(op: &Operator) -> Result<(), Error> {
    if op == &op.adjoint() {
        Ok(())
    } else {
        Err(Error::NotHermitian)
    }
}

pub fn check_positivity(op: &Operator) -> Result<(), Error> {
    check_hermiticity(op)?;
    if op.symmetric_eigenvalues().iter().all(|&e| e >= 0.) {
        Ok(())
    } else {
        Err(Error::NotPositiveSemidefinite)
    }
}

pub fn check_state(state: &State) -> Result<(), Error> {
    // if we know that the operator is positive semidefinite, surely the diagonal
    // is real, otherwise it would not be hermitian
    check_positivity(state)?;
    if state.trace().re == 1. {
        Ok(())
    } else {
        Err(Error::NotUnitaryTrace(state.trace().re))
    }
}
