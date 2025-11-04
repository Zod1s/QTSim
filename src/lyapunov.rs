use crate::utils::*;
pub(crate) use lapack;
use nalgebra as na;

pub fn lyapunovequation<D: na::Dim>(
    a: &na::OMatrix<f64, D, D>,
    q: &na::OMatrix<f64, D, D>,
) -> SolverResult<na::OMatrix<f64, D, D>>
where
    D: na::DimSub<na::Const<1>>,
    na::DefaultAllocator: na::allocator::Allocator<D>,
    na::DefaultAllocator: na::allocator::Allocator<D, D>,
    na::DefaultAllocator: na::allocator::Allocator<<D as na::DimSub<na::Const<1>>>::Output>,
    na::DefaultAllocator: na::allocator::Allocator<D, <D as na::DimSub<na::Const<1>>>::Output>,
{
    if !a.complex_eigenvalues().iter().all(|eig| eig.re < 0.) {
        return Err(SolverError::NonAsymptoticallyStableError);
    } else if !(&q.transpose() == q) || !q.symmetric_eigenvalues().iter().all(|eig| eig < &0.) {
        return Err(SolverError::NonPositiveDefiniteError);
    }

    let mut p = q.clone_owned();

    let schura = a.clone_owned().schur();
    let (v, t) = schura.unpack();
    p = v.transpose() * q * &v;

    let mut scale = 1.;
    let mut info = 0;
    unsafe {
        lapack::dtrsyl(
            'T' as u8,
            'N' as u8,
            &[1],
            3,
            3,
            t.as_slice(),
            3,
            t.as_slice(),
            3,
            p.as_mut_slice(),
            3,
            &mut [scale],
            &mut info,
        )
    };
    lapackerror(info)?;
    Ok(&v * p * v.transpose())
}
