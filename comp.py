import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1j], [1j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

paulis = [paulix, pauliy, pauliz]
weights = [1, 1, 2]

# s1 = list(map(lambda x: np.kron(np.kron(np.kron(
#     np.kron(x, np.eye(2)), np.eye(2)), np.eye(2)), np.eye(2)), paulis))
# s2 = list(map(lambda x: np.kron(np.kron(np.kron(
#     np.kron(np.eye(2), x), np.eye(2)), np.eye(2)), np.eye(2)), paulis))
# s3 = list(map(lambda x: np.kron(np.kron(np.kron(
#     np.kron(np.eye(2), np.eye(2)), x), np.eye(2)), np.eye(2)), paulis))
# s4 = list(map(lambda x: np.kron(np.kron(np.kron(
#     np.kron(np.eye(2), np.eye(2)), np.eye(2)), x), np.eye(2)), paulis))
# s5 = list(map(lambda x: np.kron(np.kron(np.kron(
#     np.kron(np.eye(2), np.eye(2)), np.eye(2)), np.eye(2)), x), paulis))
s1 = list(map(lambda x: np.kron(np.kron(x, np.eye(2)), np.eye(2)), paulis))
s2 = list(map(lambda x: np.kron(np.kron(np.eye(2), x), np.eye(2)), paulis))
s3 = list(map(lambda x: np.kron(np.kron(np.eye(2), np.eye(2)), x), paulis))

H = np.zeros((8, 8), dtype=np.complex128)

for p1, p2, w in zip(s1, s2, weights):
    H += p1 @ p2 * w
for p1, p2, w in zip(s2, s3, weights):
    H += p1 @ p2 * w
# for p1, p2, w in zip(s3, s4, weights):
#     H -= p1 @ p2 * w
# for p1, p2, w in zip(s4, s5, weights):
#     H -= p1 @ p2 * w
for p1, p2, w in zip(s3, s1, weights):
    H += p1 @ p2 * w

print(H)

# F0 = np.ones((8, 8)) - np.eye(8)
# F0 = 4 * F0
# F0 = np.diag([1, 1, 1, 1, 1, 1, 1], k=1)
# F0 = F0 + np.conj(F0.T)
eigs, eigvs = np.linalg.eigh(H)
print(eigs)
print(eigvs)
# print(F0)
# F0 = eigvs @ F0 @ np.conj(eigvs.T)
# L = eigvs @ np.diag([-4, -4, 4, 4, 4, 4, 4, 4]) @ np.conj(eigvs.T)
# eigsL, eigvsL = np.linalg.eigh(L)
# print(eigvs)
# print()
# print(eigvsL)
# print()
# func = np.vectorize(lambda x: 0 if np.abs(x) < 1e-12 else x)
# np.set_printoptions(precision=2)
# print(func(np.conj(eigvs.T) @ H @ eigvs))
# print(func(np.conj(eigvsL.T) @ H @ eigvsL))
#
# I = np.eye(8)
#
# A = (
#     -1j * (np.kron(I, H + F0) - np.kron((H + F0).T, I))
#     + np.kron(np.conj(L), L)
#     - 0.5 * (np.kron(I, np.conj(L.T) @ L) + np.kron(L.T @ np.conj(L), I))
# )

# print(A.shape)
# print(np.linalg.matrix_rank(A))
# print(A @ np.reshape(np.eye(8), newshape=64))
