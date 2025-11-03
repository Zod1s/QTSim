use crate::utils::*;
use nalgebra as na;

pub fn lyapunov2x2(a: &na::Matrix2<f64>, q: &na::Matrix2<f64>) -> SolverResult<na::Matrix2<f64>> {
    let mut pivotingmatrix = na::Matrix3::<f64>::zeros();
    pivotingmatrix[(0, 0)] = a[(0, 0)];
    pivotingmatrix[(1, 1)] = a[(0, 0)] + a[(1, 1)];
    pivotingmatrix[(2, 2)] = a[(1, 1)];

    pivotingmatrix[(0, 1)] = a[(0, 1)];
    pivotingmatrix[(1, 0)] = a[(1, 0)];
    pivotingmatrix[(1, 2)] = a[(0, 1)];
    pivotingmatrix[(2, 1)] = a[(1, 0)];

    let mut knownterms = na::vector![q[(0, 0)] / 2., q[(0, 1)], q[(1, 1)] / 2.];
    let mut ipsv = 0;
    let mut jpsv = 0;
    let mut jpivot = vec![0, 0, 0];
    let mut tmp = vec![0., 0., 0.];

    for k in 0..2 {
        let mut xmax = 0.;
        for i in k..3 {
            for j in k..3 {
                if pivotingmatrix[(i, j)].abs() >= xmax {
                    xmax = pivotingmatrix[(i, j)].abs();
                    ipsv = i;
                    jpsv = j;
                }
            }
        }
        if ipsv != k {
            let temprow = pivotingmatrix.row(ipsv).clone_owned();
            let tempkrow = pivotingmatrix.row(k).clone_owned();
            pivotingmatrix.set_row(ipsv, &tempkrow);
            pivotingmatrix.set_row(k, &temprow);
            (knownterms[k], knownterms[ipsv]) = (knownterms[ipsv], knownterms[k]);
        }
        if jpsv != k {
            let tempcolumn = pivotingmatrix.column(ipsv).clone_owned();
            let tempkcolumn = pivotingmatrix.column(k).clone_owned();
            pivotingmatrix.set_column(ipsv, &tempkcolumn);
            pivotingmatrix.set_column(k, &tempcolumn);
        }
        jpivot[k] = jpsv;

        for i in k + 1..3 {
            pivotingmatrix[(i, k)] = pivotingmatrix[(i, k)] / pivotingmatrix[(k, k)];
            knownterms[i] = knownterms[i] - pivotingmatrix[(i, k)] * knownterms[k];
            for j in k + 1..3 {
                pivotingmatrix[(i, j)] =
                    pivotingmatrix[(i, j)] - pivotingmatrix[(i, k)] * pivotingmatrix[(k, j)];
            }
        }
    }

    for i in 0..3 {
        let k = 2 - i;
        let temp = 1. / pivotingmatrix[(k, k)];
        tmp[k] = knownterms[k] * temp;

        for j in k + 1..3 {
            tmp[k] = tmp[k] - temp * pivotingmatrix[(k, j)] * tmp[j];
        }
    }

    for i in 0..2 {
        if jpivot[2 - i] != 2 - i {
            let temp = tmp[2 - i];
            tmp[2 - i] = tmp[jpivot[2 - i]];
            tmp[jpivot[2 - i]] = temp;
        }
    }

    let mut x = na::Matrix2::zeros();
    x[(0, 0)] = tmp[0];
    x[(0, 1)] = tmp[1];
    x[(1, 0)] = tmp[1];
    x[(1, 1)] = tmp[2];

    Ok(x)
}

pub fn lyapunovnxn<const D: usize>(
    a: &na::OMatrix<f64, na::Const<D>, na::Const<D>>,
    q: &mut na::OMatrix<f64, na::Const<D>, na::Const<D>>,
) -> SolverResult<na::OMatrix<f64, na::Const<D>, na::Const<D>>>
where
    na::Const<D>: na::Dim,
    na::DefaultAllocator: na::allocator::Allocator<na::Const<D>, na::Const<D>>,
{
    let mut lnext = D;
    let mut knext = 0;

    let mut mink1n = 0;
    let mut mink2n = 0;
    let mut minl1n = 0;
    let mut minl2n = 0;

    let mut l1 = 0;
    let mut l2 = 0;
    let mut k1 = 0;
    let mut k2 = 0;

    let mut vec = vec![vec![0., 0.], vec![0., 0.]];
    let mut x = vec![vec![0., 0.], vec![0., 0.]];

    for l in (0..D).rev() {
        if l > lnext {
            continue;
        }

        l1 = l;
        l2 = l;

        if l > 0 {
            if a[(l, l - 1)] != 0. {
                l1 -= 1;
            }
            lnext -= 1;
        }

        minl1n = (l1 + 1).min(D);
        minl2n = (l2 + 1).min(D);
        knext = l;

        for k in (0..l).rev() {
            if k >= knext {
                continue;
            }

            k1 = k;
            k2 = k;

            if k > 0 {
                if a[(k, k - 1)] != 0. {
                    k1 = k1 - 1;
                }
                knext = k1 - 1;
            }
            mink1n = (k1 + 1).min(D);
            mink2n = (k2 + 1).min(D);

            if l1 == l2 && k1 == k2 {
                vec[0][0] = q[(k1, l1)]
                    - a.view((k1, mink1n), (D - k1, 1))
                        .dot(&q.view((mink1n, l1), (1, D - k1)))
                    + q.view((k1, minl1n), (D - l1, 1))
                        .dot(&a.view((l1, minl1n), (1, D - l1)));

                let a11 = a[(k1, k1)] + a[(l1, l1)];

                x[0][0] = vec[0][0] / a11;
                q[(k1, l1)] = x[0][0];
                if k1 != l1 {
                    q[(l1, k1)] = x[0][0];
                }
            } else if l1 == l2 && k1 != k2 {
                vec[0][0] = q[(k1, l1)]
                    - a.view((k1, mink2n), (D - k2, 1))
                        .dot(&q.view((mink2n, l1), (1, D - k2)))
                    + q.view((k1, minl2n), (D - l2, 1))
                        .dot(&a.view((l1, minl2n), (1, D - l2)));

                vec[1][0] = q[(k2, l1)]
                    - a.view((k2, mink2n), (D - k2, 1))
                        .dot(&q.view((mink2n, l1), (1, D - k2)))
                    + q.view((k2, minl2n), (D - l2, 1))
                        .dot(&a.view((l1, minl2n), (1, D - l2)));

                let a11 = a[(k1, k1)] + a[(l1, l1)];

                x[0][0] = vec[0][0] / a11;
                q[(k1, l1)] = x[0][0];
                if k1 != l1 {
                    q[(l1, k1)] = x[0][0];
                }
            }
        }
    }

    panic!("")
}
