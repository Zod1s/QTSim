use lazy_static::lazy_static;
pub use nalgebra as na;
use nalgebra::Const;
use std::ops::{Add, Mul, Sub};
use thiserror::Error;

type TMul<T> = <T as Mul>::Output;

pub type State<D> = na::OMatrix<na::Complex<f64>, D, D>;
pub type Operator<D> = State<D>;
pub type SolverResult<T> = Result<T, SolverError>;

// lazy_static! {
//     pub static ref PAULI_X: Operator = na::dmatrix![
//         na::Complex::ZERO,
//         na::Complex::ONE;
//         na::Complex::ONE,
//         na::Complex::ZERO
//     ];
//     pub static ref PAULI_Y: Operator<2> = na::dmatrix![
//         na::Complex::ZERO,
//         na::Complex::new(0., -1.);
//         na::Complex::ZERO,
//         na::Complex::I
//     ];
//     pub static ref PAULI_Z: Operator<2> = na::dmatrix![
//         na::Complex::ONE,
//         na::Complex::ZERO;
//         na::Complex::ZERO,
//         na::Complex::new(-1., 0.)
//     ];
// }

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

pub fn to_bloch<D>(rho: &State<D>) -> SolverResult<na::SVector<f64, 3>>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    if rho.shape() != (2, 2) {
        Err(SolverError::ShapeError(rho.shape(), (2, 2)))
    } else {
        Ok(na::vector![
            2. * rho[(0, 1)].re,
            2. * rho[(1, 0)].im,
            (rho[(0, 0)] - rho[(1, 1)]).re,
        ])
    }
}

#[derive(Copy, Clone, Debug, Error)]
pub enum SolverError {
    #[error("Error, found shape {0:?}, expected shape {1:?}")]
    ShapeError((usize, usize), (usize, usize)),
    #[error("Error, the given matrix is not square, with {0} rows and {1} cols")]
    NotSquareError(usize, usize),
    #[error("Error, the given time step {0} is not positive")]
    NotPositiveDt(f64),
    #[error("Error, the given measurement efficiency {0} is not valid since it is not in the interval [0, 1]")]
    InvalidEfficiency(f64),
    #[error("Error, the given matrix is not hermitian")]
    NotHermitian,
    #[error("Error, the given matrix is not positive semidefinite")]
    NotPositiveSemidefinite,
    #[error("Error, the given matrix has trace {0}, which is different from 1")]
    NotUnitaryTrace(f64),
    #[error("Error, the final time of the simulation cannot be negative")]
    NegativeFinalTime,
    #[error("Error, the noise operators have incompatible noise shapes")]
    IncompatibleNoiseShapes,
    #[error("Error, some of the noise operators are not square")]
    NotSquareNoises,
    #[error("Error, the number of etas, {0}, is different from the number of ls, {1}")]
    NoiseEfficiencyMismatch(usize, usize),
    #[error("Error encountered while plotting")]
    PlotError,
    #[error("Error, the iterator was empty")]
    EmptyIterator,
}

// impl<E: std::error::Error + Send + Sync> From<DrawingAreaErrorKind<E>> for SolverError {
//     fn from(value: DrawingAreaErrorKind<E>) -> Self {
//         eprintln!("{}", value);
//         Self::PlotError
//     }
// }

pub fn check_hermiticity<D>(op: &Operator<D>) -> SolverResult<()>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
{
    if op == &op.adjoint() {
        Ok(())
    } else {
        Err(SolverError::NotHermitian)
    }
}

pub fn check_positivity<D>(op: &Operator<D>) -> SolverResult<()>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
{
    check_hermiticity(op)?;
    if op.symmetric_eigenvalues().iter().all(|&e| e >= 0.) {
        Ok(())
    } else {
        Err(SolverError::NotPositiveSemidefinite)
    }
}

pub fn check_state<D>(state: &State<D>) -> SolverResult<()>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
{
    // if we know that the operator is positive semidefinite, surely the diagonal
    // is real, otherwise it would not be hermitian
    check_positivity(state)?;
    if state.trace().re == 1. {
        Ok(())
    } else {
        Err(SolverError::NotUnitaryTrace(state.trace().re))
    }
}
