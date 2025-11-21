pub(crate) use nalgebra as na;
use plotpy::StrError;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Error;
use std::{
    num::ParseIntError,
    ops::{Add, Mul, Sub},
};
use thiserror::Error;

type TMul<T> = <T as Mul>::Output;

pub type Operator<D> = na::OMatrix<na::Complex<f64>, D, D>;
pub type State<D> = Operator<D>;
pub type SolverResult<T> = Result<T, SolverError>;
pub type BlochVector = na::SVector<f64, 3>;
pub type QubitState = State<na::Const<2>>;
pub type QubitOperator = Operator<na::Const<2>>;

const TOL: f64 = 1e-10_f64;

const ZERO: na::Complex<f64> = na::Complex::ZERO;
const ONE: na::Complex<f64> = na::Complex::ONE;
const SQRT2: na::Complex<f64> = na::Complex::new(1. / std::f64::consts::SQRT_2, 0.);
const I: na::Complex<f64> = na::Complex::I;
const ISQRT2: na::Complex<f64> = na::Complex::new(0., 1. / std::f64::consts::SQRT_2);
const MONE: na::Complex<f64> = na::Complex::new(-1., 0.);
const MSQRT2: na::Complex<f64> = na::Complex::new(-1. / std::f64::consts::SQRT_2, 0.);
const MI: na::Complex<f64> = na::Complex::new(0., -1.);
const MISQRT2: na::Complex<f64> = na::Complex::new(0., -1. / std::f64::consts::SQRT_2);
const FRAC_1_SQRT_3: f64 = 0.577350269189625764509148780501957456_f64;

pub const PAULI_X: QubitOperator = na::Matrix2::new(ZERO, ONE, ONE, ZERO);

pub const PAULI_Y: QubitOperator = na::Matrix2::new(ZERO, MI, I, ZERO);

pub const PAULI_Z: QubitOperator = na::Matrix2::new(ONE, ZERO, ZERO, MONE);

pub const GELLMANN1: Operator<na::Const<3>> =
    na::Matrix3::new(ZERO, SQRT2, ZERO, SQRT2, ZERO, ZERO, ZERO, ZERO, ZERO);
pub const GELLMANN2: Operator<na::Const<3>> =
    na::Matrix3::new(ZERO, MISQRT2, ZERO, ISQRT2, ZERO, ZERO, ZERO, ZERO, ZERO);
pub const GELLMANN3: Operator<na::Const<3>> =
    na::Matrix3::new(SQRT2, ZERO, ZERO, ZERO, MSQRT2, ZERO, ZERO, ZERO, ZERO);
pub const GELLMANN4: Operator<na::Const<3>> =
    na::Matrix3::new(ZERO, ZERO, SQRT2, ZERO, ZERO, ZERO, SQRT2, ZERO, ZERO);
pub const GELLMANN5: Operator<na::Const<3>> =
    na::Matrix3::new(ZERO, ZERO, MISQRT2, ZERO, ZERO, ZERO, ISQRT2, ZERO, ZERO);
pub const GELLMANN6: Operator<na::Const<3>> =
    na::Matrix3::new(ZERO, ZERO, ZERO, ZERO, ZERO, SQRT2, ZERO, SQRT2, ZERO);
pub const GELLMANN7: Operator<na::Const<3>> =
    na::Matrix3::new(ZERO, ZERO, ZERO, ZERO, ZERO, MISQRT2, ZERO, ISQRT2, ZERO);
pub const GELLMANN8: Operator<na::Const<3>> = na::Matrix3::new(
    na::Complex::new(FRAC_1_SQRT_3 / std::f64::consts::SQRT_2, 0.),
    ZERO,
    ZERO,
    ZERO,
    na::Complex::new(FRAC_1_SQRT_3 / std::f64::consts::SQRT_2, 0.),
    ZERO,
    ZERO,
    ZERO,
    na::Complex::new(-FRAC_1_SQRT_3 * std::f64::consts::SQRT_2, 0.),
);

pub const GELLMANNMATRICES: [Operator<na::Const<3>>; 8] = [
    GELLMANN1, GELLMANN2, GELLMANN3, GELLMANN4, GELLMANN5, GELLMANN6, GELLMANN7, GELLMANN8,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LastSet {
    NotSet,
    LeGammaSet,
    GeGamma2Set,
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

pub fn random_complex_vector<D>(seed: Option<u64>) -> na::OVector<na::Complex<f64>, D>
where
    D: na::Dim + na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<D>,
{
    let mut rng = if let Some(seed) = seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::rng())
    };

    na::OVector::from_fn(|_, _| {
        na::Complex::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0))
    })
}

pub fn random_unit_complex_vector<D>(seed: Option<u64>) -> na::OVector<na::Complex<f64>, D>
where
    D: na::Dim + na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<D>,
{
    random_complex_vector::<D>(seed).normalize()
}

pub fn random_pure_state<D>(seed: Option<u64>) -> State<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>> + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator: na::allocator::Allocator<na::Const<1>, D>,
    na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
{
    let x = random_unit_complex_vector::<D>(seed);
    let x = &x * x.adjoint();

    let mut decomp = x.symmetric_eigen();
    decomp.eigenvalues = decomp.eigenvalues.map(|e| e.max(0.));
    let x = decomp.recompose();
    for el in x.diagonal().iter_mut() {
        *el = na::Complex::new(el.re, 0.);
    }
    x.scale(1. / x.trace().re)
}

pub fn random_vector_3(seed: Option<u64>) -> na::SVector<f64, 3> {
    let mut rng = if let Some(seed) = seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::rng())
    };
    na::Vector3::new(
        rng.random_range(-1.0..1.0),
        rng.random_range(-1.0..1.0),
        rng.random_range(-1.0..1.0),
    )
}

pub fn random_blochvector(seed: Option<u64>) -> BlochVector {
    loop {
        let vec = random_vector_3(seed);
        if vec.norm() <= 1. {
            return vec;
        }
    }
}

pub fn random_pure_blochvector(seed: Option<u64>) -> BlochVector {
    let vec = random_vector_3(seed);
    vec.normalize()
}

pub fn random_pure_qubitstate(seed: Option<u64>) -> QubitState {
    from_bloch(&random_pure_blochvector(seed)).expect("It should already have norm 1")
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

pub fn random_qubit_state(seed: Option<u64>) -> QubitState {
    from_bloch(&random_blochvector(seed))
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
    NotUnitTrace(f64),
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
    #[error("IO Error")]
    IoError,
    #[error("Non asymptotically stable matrix")]
    NonAsymptoticallyStableError,
    #[error("Non positive definite matrix")]
    NonPositiveDefiniteError,
    #[error("Lapack error: {0}")]
    LapackError(i32),
}

impl From<StrError> for SolverError {
    fn from(value: StrError) -> Self {
        Self::PlotPyError(value)
    }
}

impl From<std::io::Error> for SolverError {
    fn from(value: std::io::Error) -> Self {
        eprintln!("{}", value);
        Self::IoError
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
        Err(SolverError::NotUnitTrace(state.trace().re))
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

pub fn lindbladian<D>(h: &Operator<D>, l: &Operator<D>, x: &Operator<D>) -> Operator<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    hamiltonian_term(h, x) + measurement_term(l, x)
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

fn sqrthm<D>(rho: &Operator<D>) -> Operator<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>> + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
{
    let mut decomp = rho.clone().symmetric_eigen();
    decomp.eigenvalues = decomp
        .eigenvalues
        .map(|e| if e > TOL { f64::sqrt(e) } else { 0. });
    decomp.recompose()
}

pub fn fidelity<D>(rho: &State<D>, sigma: &State<D>) -> f64
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>> + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
{
    let rhosqrt = sqrthm(rho);
    sqrthm(&(&rhosqrt * sigma * &rhosqrt)).trace().re.powi(2)
}

pub fn rouchonstep<D>(
    dt: f64,
    rho: &State<D>,
    h: &Operator<D>,
    l: &Operator<D>,
    dw: f64,
) -> State<D>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>> + std::marker::Copy,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
{
    let id = Operator::<D>::identity();
    let fst = (h * na::Complex::I + l.adjoint() * l.scale(0.5)).scale(dt);
    let snd = l.scale(((l + l.adjoint()) * rho).trace().re * dt + dw);
    let thd = (l * l).scale(dw.powi(2) - dt).scale(0.5);

    let m = id - fst + snd + thd;

    let num = &m * rho * m.adjoint();
    num.scale(1. / num.trace().re) - rho
}

pub fn veclindblad<D>(h: &Operator<D>, l: &Operator<D>) -> Operator<na::DimProd<D, D>>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>> + std::marker::Copy + na::DimMul<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator:
        na::allocator::Allocator<<D as na::DimMul<D>>::Output, <D as na::DimMul<D>>::Output>,
{
    let id = Operator::<D>::identity();
    -(id.kronecker(h) - h.conjugate().kronecker(&id)) * na::Complex::I + l.conjugate().kronecker(&l)
        - (id.kronecker(&(l.adjoint() * l)) + (l.adjoint() * l).transpose().kronecker(&id))
            .scale(0.5)
}

pub fn veclindbladheis<D>(h: &Operator<D>, l: &Operator<D>) -> Operator<na::DimProd<D, D>>
where
    D: na::Dim + na::DimName + na::DimSub<na::Const<1>> + std::marker::Copy + na::DimMul<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator:
        na::allocator::Allocator<<D as na::DimMul<D>>::Output, <D as na::DimMul<D>>::Output>,
{
    let id = Operator::<D>::identity();
    (id.kronecker(h) - h.conjugate().kronecker(&id)) * na::Complex::I
        + l.transpose().kronecker(&l.adjoint())
        - (id.kronecker(&(l.adjoint() * l)) + (l.adjoint() * l).transpose().kronecker(&id))
            .scale(0.5)
}

pub fn lapackerror(info: i32) -> SolverResult<()> {
    if info >= 0 {
        Ok(())
    } else {
        Err(SolverError::LapackError(info))
    }
}

pub fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);

    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.
    } else {
        v[mid]
    }
}
