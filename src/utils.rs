pub(crate) use nalgebra as na;
use plotpy::StrError;
use rand::prelude::*;
use std::{
    num::ParseIntError,
    ops::{Add, Mul, Sub},
};
use thiserror::Error;

type TMul<T> = <T as Mul>::Output;

pub type State<D> = na::OMatrix<na::Complex<f64>, D, D>;
pub type Operator<D> = State<D>;
pub type SolverResult<T> = Result<T, SolverError>;
pub type BlochVector = na::SVector<f64, 3>;
pub type QubitState = State<na::Const<2>>;
pub type QubitOperator = Operator<na::Const<2>>;

// const EPSILON: f64 = 1e-10;

pub const PAULI_X: QubitOperator = na::Matrix2::new(
    na::Complex::ZERO,
    na::Complex::ONE,
    na::Complex::ONE,
    na::Complex::ZERO,
);

pub const PAULI_Y: QubitOperator = na::Matrix2::new(
    na::Complex::ZERO,
    na::Complex::new(0.0, -1.0),
    na::Complex::I,
    na::Complex::ZERO,
);

pub const PAULI_Z: QubitOperator = na::Matrix2::new(
    na::Complex::ONE,
    na::Complex::ZERO,
    na::Complex::ZERO,
    na::Complex::new(-1.0, 0.0),
);

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

pub fn kroneckerdelta<A: Eq>(x: &A, y: &A) -> f64 {
    if x == y {
        1.
    } else {
        0.
    }
}

pub fn to_bloch(rho: &QubitState) -> SolverResult<BlochVector> {
    let bloch = na::vector![
        2. * rho[(0, 1)].re,
        2. * rho[(1, 0)].im,
        (rho[(0, 0)] - rho[(1, 1)]).re,
    ];
    if bloch.norm() > 1. {
        Err(SolverError::BlochNormError(bloch.norm()))
    } else {
        Ok(bloch)
    }
}

pub fn to_bloch_unchecked(rho: &QubitState) -> BlochVector {
    let bloch = na::vector![
        2. * rho[(0, 1)].re,
        2. * rho[(1, 0)].im,
        (rho[(0, 0)] - rho[(1, 1)]).re,
    ];
    bloch
}

pub fn random_complex_vector<const D: usize>() -> na::SVector<na::Complex<f64>, D> {
    let mut rng = rand::rng();
    na::SVector::from_fn(|_, _| {
        na::Complex::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
    })
}

pub fn random_unit_complex_vector<const D: usize>() -> na::SVector<na::Complex<f64>, D> {
    random_complex_vector::<D>().normalize()
}

pub fn random_vector_3() -> na::SVector<f64, 3> {
    let mut rng = rand::rng();
    na::Vector3::new(
        rng.random_range(-1.0..1.0),
        rng.random_range(-1.0..1.0),
        rng.random_range(-1.0..1.0),
    )
}

pub fn random_blochvector() -> BlochVector {
    loop {
        let vec = random_vector_3();
        if vec.norm() <= 1. {
            return vec;
        }
    }
}

pub fn random_pure_blochvector() -> BlochVector {
    let vec = random_vector_3();
    vec.normalize()
}

pub fn random_pure_qubitstate() -> QubitState {
    from_bloch(&random_pure_blochvector()).expect("It should already have norm 1")
}

pub fn from_bloch(bloch: &BlochVector) -> SolverResult<QubitState> {
    if bloch.norm() > 1. {
        Err(SolverError::BlochNormError(bloch.norm()))
    } else {
        Ok((QubitOperator::identity()
            + PAULI_X.scale(bloch[0])
            + PAULI_Y.scale(bloch[1])
            + PAULI_Z.scale(bloch[2]))
        .scale(0.5))
    }
}

pub fn from_bloch_unchecked(bloch: &BlochVector) -> QubitState {
    (QubitOperator::identity()
        + PAULI_X.scale(bloch[0])
        + PAULI_Y.scale(bloch[1])
        + PAULI_Z.scale(bloch[2]))
    .scale(0.5)
}

pub fn random_qubit_state() -> QubitState {
    from_bloch(&random_blochvector())
        .expect("Cannot have norm larger than 1 by construction, this should never fail")
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
    #[error("Error, a Bloch vector must have 3 components, found {0}")]
    BlochSizeError(usize),
    #[error("Error, a Bloch vector must have norm less than 1, the given one has norm {0}")]
    BlochNormError(f64),
    #[error("{0}")]
    PlotPyError(&'static str),
}

impl From<StrError> for SolverError {
    fn from(value: StrError) -> Self {
        Self::PlotPyError(value)
    }
}

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

pub fn hamiltonian_term<D>(h: &Operator<D>, rho: &State<D>) -> Operator<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    -commutator(h, rho) * na::Complex::I
}

pub fn measurement_term<D>(l: &Operator<D>, rho: &State<D>) -> Operator<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    l * rho * l.adjoint() - anticommutator(&(l.adjoint() * l), rho).scale(0.5)
}

pub fn noise_term<D>(l: &Operator<D>, rho: &State<D>) -> Operator<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    l * rho + rho * l.adjoint() - rho * ((l + l.adjoint()) * rho).trace()
}

pub fn corrected_hamiltonian<D>(h: &Operator<D>, l: &Operator<D>, f: &Operator<D>) -> Operator<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    h + (f * l + l.adjoint() * f).scale(0.5)
}

pub fn check_qubit_feeback(h: &QubitOperator, l: &QubitOperator, f: &QubitOperator) {
    let rhod = na::Matrix2::new(0., 0., 0., 1.).cast::<na::Complex<f64>>();
    let comm_check = commutator(&rhod, &(l + l.adjoint()));
    let lif = l - f * na::Complex::I;
    let htot = h + (f * l + l.adjoint() * f).scale(0.5);
    let ham_check = htot[(1, 0)] * na::Complex::I - 0.5 * lif[(1, 1)].conj() * lif[(1, 0)];

    println!("[rhod, L + L^\\dag] = {}", comm_check);
    println!("L - iF: {}", lif);
    println!("H + 0.5(FL + L^\\dag F): {}", htot);

    println!("lp: {}", lif[(1, 0)]);
    println!("lq: {}", lif[(0, 1)]);
    println!("ihp - 0.5 * ls^\\dag lp: {}", ham_check);

    let feed = lif[(1, 0)] != na::Complex::ZERO
        && lif[(0, 1)] == na::Complex::ZERO
        && ham_check == na::Complex::ZERO
        && comm_check != QubitOperator::zeros();

    println!("Valid feedback: {feed}")
}
