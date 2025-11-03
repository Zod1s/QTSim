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
