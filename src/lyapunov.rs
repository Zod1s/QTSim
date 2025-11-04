use crate::utils::*;
pub(crate) use lapack;
use nalgebra as na;

pub fn lyapunovequation() {}

// pub fn lyapunov2x2(a: &na::Matrix2<f64>, q: &na::Matrix2<f64>) -> SolverResult<na::Matrix2<f64>> {
//     let mut pivotingmatrix = na::Matrix3::<f64>::zeros();
//     pivotingmatrix[(0, 0)] = a[(0, 0)];
//     pivotingmatrix[(1, 1)] = a[(0, 0)] + a[(1, 1)];
//     pivotingmatrix[(2, 2)] = a[(1, 1)];
//
//     pivotingmatrix[(0, 1)] = a[(0, 1)];
//     pivotingmatrix[(1, 0)] = a[(1, 0)];
//     pivotingmatrix[(1, 2)] = a[(0, 1)];
//     pivotingmatrix[(2, 1)] = a[(1, 0)];
//
//     let mut knownterms = na::vector![q[(0, 0)] / 2., q[(0, 1)], q[(1, 1)] / 2.];
//     let mut ipsv = 0;
//     let mut jpsv = 0;
//     let mut jpivot = vec![0, 0, 0];
//     let mut tmp = vec![0., 0., 0.];
//
//     for k in 0..2 {
//         let mut xmax = 0.;
//         for i in k..3 {
//             for j in k..3 {
//                 if pivotingmatrix[(i, j)].abs() >= xmax {
//                     xmax = pivotingmatrix[(i, j)].abs();
//                     ipsv = i;
//                     jpsv = j;
//                 }
//             }
//         }
//         if ipsv != k {
//             let temprow = pivotingmatrix.row(ipsv).clone_owned();
//             let tempkrow = pivotingmatrix.row(k).clone_owned();
//             pivotingmatrix.set_row(ipsv, &tempkrow);
//             pivotingmatrix.set_row(k, &temprow);
//             (knownterms[k], knownterms[ipsv]) = (knownterms[ipsv], knownterms[k]);
//         }
//         if jpsv != k {
//             let tempcolumn = pivotingmatrix.column(jpsv).clone_owned();
//             let tempkcolumn = pivotingmatrix.column(k).clone_owned();
//             pivotingmatrix.set_column(jpsv, &tempkcolumn);
//             pivotingmatrix.set_column(k, &tempcolumn);
//         }
//         jpivot[k] = jpsv;
//
//         for i in k + 1..3 {
//             pivotingmatrix[(i, k)] = pivotingmatrix[(i, k)] / pivotingmatrix[(k, k)];
//             knownterms[i] = knownterms[i] - pivotingmatrix[(i, k)] * knownterms[k];
//             for j in k + 1..3 {
//                 pivotingmatrix[(i, j)] =
//                     pivotingmatrix[(i, j)] - pivotingmatrix[(i, k)] * pivotingmatrix[(k, j)];
//             }
//         }
//     }
//
//     for i in 0..3 {
//         let k = 2 - i;
//         let temp = 1. / pivotingmatrix[(k, k)];
//         tmp[k] = knownterms[k] * temp;
//
//         for j in k + 1..3 {
//             tmp[k] = tmp[k] - temp * pivotingmatrix[(k, j)] * tmp[j];
//         }
//     }
//
//     for i in 0..2 {
//         if jpivot[2 - i] != 2 - i {
//             let temp = tmp[2 - i];
//             tmp[2 - i] = tmp[jpivot[2 - i]];
//             tmp[jpivot[2 - i]] = temp;
//         }
//     }
//
//     let mut x = na::Matrix2::zeros();
//     x[(0, 0)] = tmp[0];
//     x[(0, 1)] = tmp[1];
//     x[(1, 0)] = tmp[1];
//     x[(1, 1)] = tmp[2];
//
//     Ok(x)
// }
//
// pub fn lyapunovnxn<const D: usize>(
//     a: &na::OMatrix<f64, na::Const<D>, na::Const<D>>,
//     q: &na::OMatrix<f64, na::Const<D>, na::Const<D>>,
// ) -> SolverResult<na::OMatrix<f64, na::Const<D>, na::Const<D>>>
// where
//     na::Const<D>: na::Dim,
//     na::DefaultAllocator: na::allocator::Allocator<na::Const<D>, na::Const<D>>,
// {
//     let mut p = q.clone_owned();
//
//     let mut lnext = D;
//     let mut knext;
//
//     let mut mink1n;
//     let mut mink2n;
//     let mut minl1n;
//     let mut minl2n;
//
//     let mut l1;
//     let mut l2;
//     let mut k1;
//     let mut k2;
//
//     let mut vec = na::matrix![0., 0.; 0., 0.];
//     let mut x = na::matrix![0., 0.; 0., 0.];
//     let id = na::matrix![1., 0.; 0., 1.];
//
//     for l in (1..=D).rev() {
//         if l > lnext {
//             continue;
//         }
//
//         l1 = l;
//         l2 = l;
//
//         if l > 1 {
//             if a[(l - 1, l - 2)] != 0. {
//                 l1 -= 1;
//             }
//             lnext = l1 - 1;
//         }
//
//         minl1n = (l1 + 1).min(D);
//         minl2n = (l2 + 1).min(D);
//         knext = l;
//
//         for k in (1..=l).rev() {
//             if k > knext {
//                 continue;
//             }
//
//             k1 = k;
//             k2 = k;
//
//             if k > 1 {
//                 if a[(k - 1, k - 2)] != 0. {
//                     k1 = k1 - 1;
//                 }
//                 knext = k1 - 1;
//             }
//             mink1n = (k1 + 1).min(D);
//             mink2n = (k2 + 1).min(D);
//
//             if l1 == l2 && k1 == k2 {
//                 vec[(0, 0)] = q[(k1 - 1, l1 - 1)];
//                 vec[(0, 0)] -= a
//                     .view((k1 - 1, mink1n - 1), (1, D - k1))
//                     .transpose()
//                     .dot(&q.view((mink1n - 1, l1 - 1), (D - k1, 1)));
//                 vec[(0, 0)] += q
//                     .view((k1 - 1, minl1n - 1), (1, D - l1))
//                     .transpose()
//                     .dot(&a.view((l1 - 1, minl1n - 1), (1, D - l1)).transpose());
//
//                 let a11 = a[(k1 - 1, k1 - 1)] + a[(l1 - 1, l1 - 1)];
//
//                 x[(0, 0)] = vec[(0, 0)] / a11;
//                 p[(k1 - 1, l1 - 1)] = x[(0, 0)];
//
//                 if k1 != l1 {
//                     p[(l1 - 1, k1 - 1)] = x[(0, 0)];
//                 }
//             } else if l1 == l2 && k1 != k2 {
//                 vec[(0, 0)] = q[(k1 - 1, l1 - 1)]
//                     - a.view((k1 - 1, mink2n - 1), (1, D - k2))
//                         .transpose()
//                         .dot(&q.view((mink2n - 1, l1 - 1), (D - k2, 1)))
//                     + q.view((k1 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l1 - 1, minl2n - 1), (1, D - l2)).transpose());
//
//                 vec[(1, 0)] = q[(k2 - 1, l1 - 1)]
//                     - a.view((k2 - 1, mink2n - 1), (1, D - k2))
//                         .transpose()
//                         .dot(&q.view((mink2n - 1, l1 - 1), (D - k2, 1)))
//                     + q.view((k2 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l1 - 1, minl2n - 1), (1, D - l2)).transpose());
//
//                 let bara = (a.fixed_view::<2, 2>(k1 - 1, k1 - 1) + a[(l1 - 1, l1 - 1)] * id).qr();
//                 let qmat = bara.q();
//                 let rmat = bara.r();
//                 let barb = qmat.transpose() * vec.column(0);
//                 x.set_column(
//                     0,
//                     &(rmat.try_inverse().expect("Could not invert rmat") * barb),
//                 );
//
//                 p[(k1 - 1, l1 - 1)] = x[(0, 0)];
//                 p[(k2 - 1, l1 - 1)] = x[(1, 0)];
//                 p[(l1 - 1, k1 - 1)] = x[(0, 0)];
//                 p[(l1 - 1, k2 - 1)] = x[(1, 0)];
//             } else if l1 != l2 && k1 == k2 {
//                 vec[(0, 0)] = q[(k1 - 1, l1 - 1)]
//                     - a.view((k1 - 1, mink1n - 1), (1, D - k1))
//                         .transpose()
//                         .dot(&q.view((mink1n - 1, l1 - 1), (D - k1, 1)))
//                     + q.view((k1 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l1 - 1, minl2n - 1), (D - l2, 1)));
//
//                 vec[(1, 0)] = q[(k1 - 1, l2 - 1)]
//                     - a.view((k1 - 1, mink1n - 1), (1, D - k1))
//                         .transpose()
//                         .dot(&q.view((mink1n - 1, l2 - 1), (D - k1, 1)))
//                     + q.view((k1 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l2 - 1, minl2n - 1), (D - l2, 1)));
//
//                 let bara = (a.fixed_view::<2, 2>(l1 - 1, l1 - 1) + a[(k1 - 1, k1 - 1)] * id).qr();
//                 let qmat = bara.q();
//                 let rmat = bara.r();
//                 let barb = qmat.transpose() * vec.column(0);
//                 x.set_column(
//                     0,
//                     &(rmat.try_inverse().expect("Could not invert rmat") * barb),
//                 );
//                 p[(k1 - 1, l1 - 1)] = x[(0, 0)];
//                 p[(k1 - 1, l2 - 1)] = x[(1, 0)];
//                 p[(l1 - 1, k1 - 1)] = x[(0, 0)];
//                 p[(l2 - 1, k1 - 1)] = x[(1, 0)];
//             } else {
//                 vec[(0, 0)] = q[(k1 - 1, l1 - 1)]
//                     - a.view((k1 - 1, mink2n - 1), (1, D - k2))
//                         .transpose()
//                         .dot(&q.view((mink2n - 1, l1 - 1), (D - k2, 1)))
//                     + q.view((k1 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l1 - 1, minl2n - 1), (D - l2, 1)));
//
//                 vec[(0, 1)] = q[(k1 - 1, l2 - 1)]
//                     - a.view((k1 - 1, mink2n - 1), (1, D - k2))
//                         .transpose()
//                         .dot(&q.view((mink2n - 1, l2 - 1), (D - k2, 1)))
//                     + q.view((k1 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l2 - 1, minl2n - 1), (D - l2, 1)));
//
//                 vec[(1, 0)] = q[(k2 - 1, l1 - 1)]
//                     - a.view((k2 - 1, mink2n - 1), (1, D - k2))
//                         .transpose()
//                         .dot(&q.view((mink2n - 1, l1 - 1), (D - k2, 1)))
//                     + q.view((k2 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l1 - 1, minl2n - 1), (D - l2, 1)));
//
//                 vec[(1, 1)] = q[(k2 - 1, l2 - 1)]
//                     - a.view((k2 - 1, mink2n - 1), (1, D - k2))
//                         .transpose()
//                         .dot(&q.view((mink2n - 1, l2 - 1), (D - k2, 1)))
//                     + q.view((k2 - 1, minl2n - 1), (1, D - l2))
//                         .transpose()
//                         .dot(&a.view((l2 - 1, minl2n - 1), (D - l2, 1)));
//
//                 if k1 == l1 {
//                     x = lyapunov2x2(&a.fixed_view::<2, 2>(k1 - 1, k1 - 1).clone_owned(), &vec)?;
//                     x[(1, 0)] = x[(0, 1)];
//                 } else {
//                     let mut ipsv = 0;
//                     let mut jpsv = 0;
//                     let mut tempb = na::vector![vec[(0, 0)], vec[(1, 0)], vec[(0, 1)], vec[(1, 1)]];
//                     let mut jpiv = na::vector![0, 0, 0, 0];
//                     let tl = a.fixed_view::<2, 2>(k1 - 1, k1 - 1);
//                     let tr = a.fixed_view::<2, 2>(l1 - 1, l1 - 1);
//                     let mut t16 = na::matrix![
//                         tl[(0, 0)] + tr[(0, 0)],              tl[(0, 1)],              tr[(0, 1)],                      0.;
//                                      tl[(1, 0)], tl[(1, 1)] + tr[(0, 0)],                      0.,              tr[(0, 1)];
//                                      tr[(1, 0)],                      0., tl[(0, 0)] + tr[(1, 1)],              tl[(0, 1)];
//                                              0.,              tr[(1, 0)],              tl[(1, 0)], tl[(1, 1)] + tr[(1, 1)]
//                     ];
//
//                     for i in 0..3 {
//                         let mut xmax = 0.;
//                         for ip in i..4 {
//                             for jp in i..4 {
//                                 if t16[(ip, jp)].abs() >= xmax {
//                                     xmax = t16[(ip, jp)].abs();
//                                     ipsv = ip;
//                                     jpsv = jp;
//                                 }
//                             }
//                         }
//                         if ipsv != i {
//                             let temprow = t16.row(ipsv).clone_owned();
//                             let tempkrow = t16.row(i).clone_owned();
//                             t16.set_row(ipsv, &tempkrow);
//                             t16.set_row(i, &temprow);
//                             (tempb[i], tempb[ipsv]) = (tempb[ipsv], tempb[i]);
//                         }
//                         if jpsv != i {
//                             let tempcolumn = t16.column(jpsv).clone_owned();
//                             let tempkcolumn = t16.column(i).clone_owned();
//                             t16.set_column(ipsv, &tempkcolumn);
//                             t16.set_column(i, &tempcolumn);
//                         }
//                         jpiv[i] = jpsv;
//
//                         for j in i + 1..4 {
//                             t16[(j, i)] = t16[(j, i)] / t16[(i, i)];
//                             tempb[j] = tempb[j] - t16[(j, i)] * tempb[i];
//                             for j in i + 1..4 {
//                                 t16[(j, k)] = t16[(j, k)] - t16[(j, i)] * t16[(i, k)];
//                             }
//                         }
//                     }
//                     let mut tmp = na::vector![0., 0., 0., 0.];
//                     for i in 0..4 {
//                         let k = 3 - i;
//                         let temp = 1. / t16[(k, k)];
//                         tmp[k] = tempb[k] * temp;
//                         for j in k + 1..4 {
//                             tmp[k] = tmp[k] - temp * t16[(k, j)] * tmp[j];
//                         }
//                     }
//                     for i in 0..3 {
//                         if jpiv[2 - i] != 2 - i {
//                             let temp = tmp[2 - i];
//                             tmp[2 - i] = tmp[jpiv[2 - i]];
//                             tmp[jpiv[2 - i]] = temp;
//                         }
//                     }
//
//                     x[(0, 0)] = tmp[0];
//                     x[(1, 0)] = tmp[1];
//                     x[(0, 1)] = tmp[2];
//                     x[(1, 1)] = tmp[3];
//                 }
//
//                 p[(k1 - 1, l1 - 1)] = x[(0, 0)];
//                 p[(k1 - 1, l2 - 1)] = x[(0, 1)];
//                 p[(k2 - 1, l1 - 1)] = x[(1, 0)];
//                 p[(k2 - 1, l2 - 1)] = x[(1, 1)];
//                 if k1 != l1 {
//                     p[(l1 - 1, k1 - 1)] = x[(0, 0)];
//                     p[(l1 - 1, k2 - 1)] = x[(0, 1)];
//                     p[(l2 - 1, k1 - 1)] = x[(1, 0)];
//                     p[(l2 - 1, k2 - 1)] = x[(1, 1)];
//                 }
//             }
//         }
//     }
//
//     Ok(p)
// }
