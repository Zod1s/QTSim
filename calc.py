from sympy import *
from sympy.physics.quantum.dagger import Dagger

init_printing(use_unicode=True)

f0p, f1p, hp, ls, lp, lq, lr = symbols(
    "f0p f1p hp ls lp lq lr", complex=True)

f0s, f1s, f0r, f1r, hs, hr = symbols(
    "f0s f1s f0r f1r hs hr", real=True)

t, yt, yd = symbols("t yt yd", real=True)

F0 = Matrix([
    [f0s, f0p],
    [conjugate(f0p), f0r]
])

F0 = (yt - yd * t) * F0 / t

F1 = Matrix([
    [f1s, f1p],
    [conjugate(f1p), f1r]
])

L = Matrix([
    [ls, lp],
    [lq, lr]
])

H = Matrix([
    [hs, hp],
    [conjugate(hp), hr]
])

F1 = F1.subs(f1p, I * conjugate(lq))

LiF = simplify(L - I * F1)

corr = simplify(simplify(0.5 * (F1 * L + Dagger(L) * F1)))
pprint(corr)

Htot = H + F0 + corr

# pprint(F0)
# pprint(F1)
# pprint(H + F0)
# pprint(LiF)
# pprint(corr)
# pprint(Htot)

Htotp = Htot[0, 1]
Ls = LiF[0, 0]
Lp = LiF[0, 1]

eq = I * Htotp - 0.5 * conjugate(Ls) * Lp
eq = simplify(eq)
pprint(eq)

reeq = re(eq)
imeq = im(eq)
# pprint(reeq)
# pprint(imeq)

reeqnew = reeq.subs({
    ls: 0, lp: 1, lq: 1, lr: 0, hp: 0
})

imeqnew = imeq.subs({
    ls: 0, lp: 1, lq: 1, lr: 0, hp: 0
})

pprint(reeqnew)
pprint(imeqnew)
