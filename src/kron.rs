use crate::utils::*;
use nalgebra as na;

pub fn vectorisationexample() {
    let h0 =
        na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast::<na::Complex<f64>>();
    let hc = na::Matrix3::zeros();
    let f0 = na::matrix![0., 1., 1.; 1., 0., 1.; 1., 1., 0.].cast();

    let h = h0 + hc + f0;
    let l =
        na::Matrix3::from_diagonal(&na::Vector3::new(-1.0, 2.0, 3.0)).cast::<na::Complex<f64>>();
    let id: na::Matrix3<na::Complex<f64>> = na::Matrix3::identity();

    let lindbladmatrix = veclindblad(&h, &l);
    let lindbladheismatrix = veclindbladheis(&h, &l);
    let lindbladsymmsum = lindbladmatrix + lindbladheismatrix;
    let lindbladsymm = veclindblad(&na::Matrix3::zeros(), &l.scale(f64::sqrt(2.)));

    // println!("H: {h}");
    // println!("L: {l}");
    // println!("A: {lindbladmatrix}");
    // println!("Aheis: {lindbladheismatrix}");
    println!("Asymmsum: {lindbladsymmsum}");
    println!("Asymm: {lindbladsymm}");
    // println!("Eigs: {}", lindbladmatrix.eigenvalues().unwrap());
    // println!("Eigs heis: {}", lindbladheismatrix.eigenvalues().unwrap());
    // println!("Eigs symm: {}", lindbladsymm.eigenvalues().unwrap());
}

fn veclindblad<D>(h: &Operator<D>, l: &Operator<D>) -> Operator<na::DimProd<D, D>>
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

fn veclindbladheis<D>(h: &Operator<D>, l: &Operator<D>) -> Operator<na::DimProd<D, D>>
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
