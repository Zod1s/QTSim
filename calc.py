from sympy import *
from sympy.physics.quantum.dagger import Dagger
from sympy.abc import N

init_printing(use_unicode=True)

fcp, f1p, hp, ls, lp, lq, lr = symbols("fcp f1p hp ls lp lq lr", complex=True)

fcs, f1s, fcr, f1r, hs, hr = symbols("fcs f1s fcr f1r hs hr", real=True)


# H0s = MatrixSymbol("H0s", 1, 1)
# H0p = MatrixSymbol("H0p", 1, N)
# H0r = MatrixSymbol("H0r", N, N)
#
# H0 = Matrix([[H0s, H0p], [conjugate(H0p.T), H0r]])
#
# Hcs = MatrixSymbol("Hcs", 1, 1)
# Hcp = MatrixSymbol("Hcp", 1, N)
# Hcr = MatrixSymbol("Hcr", N, N)
#
# Hc = Matrix([[Hcs, Hcp], [conjugate(Hcp.T), Hcr]])
#
# Ls = MatrixSymbol("Ls", 1, 1)
# Lp = MatrixSymbol("Lp", 1, N)
# Lq = MatrixSymbol("Lq", N, 1)
# Lr = MatrixSymbol("Lr", N, N)
#
# L = Matrix([[Ls, Lp], [Lq, Lr]])
# Ldag = Matrix([[conjugate(Ls.T), conjugate(Lq.T)], [conjugate(Lp.T), conjugate(Lr.T)]])
#
# F1s = MatrixSymbol("F1s", 1, 1)
# F1r = MatrixSymbol("F1r", N, N)
#
# F1 = Matrix([[F1s, I * conjugate(Lq.T)], [-I * Lq, F1r]])
#
# pprint(H0)
# pprint(L)
# pprint(F1)
#
# Lhat = L - I * F1
# Lhat = simplify(Lhat)
# pprint(Lhat)
#
# Hamcorr = 0.5 * (F1 * L + Ldag * F1)
# Hamcorr = simplify(Hamcorr)
# pprint(Hamcorr)
#
# Totham = simplify(H0 + Hc + Hamcorr)
#
# Hp = Totham[(0, 1)]
# Lhats = Lhat[(0, 0)]
# Lhatp = Lhat[(0, 1)]
#
# final = I * Hp - 0.5 * conjugate(Lhats.T) * Lhatp
# final = simplify(final)
# pprint(final)

# Fc = Matrix([[fcs, fcp], [conjugate(fcp), fcr]])
# F1 = Matrix([[f1s, f1p], [conjugate(f1p), f1r]])

paulix = Matrix([[0, 1], [1, 0]])
pauliy = Matrix([[0, -I], [I, 0]])
pauliz = Matrix([[1, 0], [0, -1]])

pprint(pauliz + 0.5 * (-pauliy * paulix - paulix * pauliy))
pprint(paulix + I * pauliy)

# H = pauliz
# L = pauliz + 0.5 * paulix
# H = paulix * H * paulix
# L = paulix * L * paulix
#
# pprint(H)
# pprint(L)
#
# F1 = F1.subs(f1p, I * conjugate(L[1, 0]))
#
# LiF = simplify(L - I * F1)
#
# corr = simplify(simplify(0.5 * (F1 * L + Dagger(L) * F1)))
# pprint(F1)
# pprint(LiF)
# pprint(corr)
#
# Htot = simplify(H + corr + Fc)
#
# Htotp = Htot[0, 1]
# Ls = LiF[0, 0]
# Lp = LiF[0, 1]
#
# eq = I * Htotp - 0.5 * conjugate(Ls) * Lp
# eq = simplify(eq)
# pprint(eq)
#
# x, y = symbols("x y", real=True)
# f = atan(y / x)
# g = simplify(Matrix([f]).jacobian([x, y]))
# h = simplify(g.T.jacobian([x, y]))
# dzdz = Matrix(
#     [
#         [x**2, x * y],
#         [x * y, y**2],
#     ]
# )

# f = sqrt(x**2 + y**2)
# g = simplify(Matrix([f]).jacobian([x, y]))
# h = simplify(g.T.jacobian([x, y]))
# pprint(f)
# pprint(g)
# pprint(h)
# pprint(dzdz)
# pprint(simplify(h * dzdz))

# a, b, c, d = symbols("a b c d", real=True)
# nx, ny, nz = symbols("nx ny nz", real=True)
#
# paulix = Matrix([[0, 1], [1, 0]])
# pauliy = Matrix([[0, -I], [I, 0]])
# pauliz = Matrix([[1, 0], [0, -1]])
#
# rho = 0.5 * (eye(2) + nx * paulix + ny * pauliy + nz * pauliz)
# H0 = a * eye(2) + b * pauliz
# L = c * eye(2) + d * pauliz
#
# ham = simplify(-I * (H0 * rho - rho * H0))
# lrhol = simplify(L * rho * Dagger(L))
# llrho = simplify(Dagger(L) * L * rho + rho * Dagger(L) * L)

# pprint(ham)
# pprint(lrhol)
# pprint(llrho)
# pprint(simplify(ham + lrhol - 0.5 * llrho))
#
# g = L * rho + rho * Dagger(L) - Trace((L + Dagger(L)) * rho) * rho
# g = simplify(g)
# pprint(simplify(g))

# rho0 = Matrix([[1, 0], [0, 0]])
# rho1 = Matrix([[0, 0], [0, 1]])

# lv0 = Trace(
#     I * (H0 * rho - rho * H0) * rho0
#     - L * rho * Dagger(L) * rho0
#     + 0.5 * (Dagger(L) * L * rho + rho * Dagger(L) * L) * rho0
# )
#
# lv0 = simplify(simplify(lv0))
#
# lv1 = Trace(
#     I * (H0 * rho - rho * H0) * rho1
#     - L * rho * Dagger(L) * rho1
#     + 0.5 * (Dagger(L) * L * rho + rho * Dagger(L) * L) * rho1
# )
#
# lv1 = simplify(simplify(lv1))
#
# pprint(lv0)
# pprint(lv1)
