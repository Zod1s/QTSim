import numpy as np
# np.set_printoptions(threshold=np.inf)

paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1j], [1j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

paulis = [paulix, pauliy, pauliz]
weights = [0.5, 0.5, 3]

s1 = list(map(lambda x: np.kron(np.kron(np.kron(
    np.kron(x, np.eye(2)), np.eye(2)), np.eye(2)), np.eye(2)), paulis))
s2 = list(map(lambda x: np.kron(np.kron(np.kron(
    np.kron(np.eye(2), x), np.eye(2)), np.eye(2)), np.eye(2)), paulis))
s3 = list(map(lambda x: np.kron(np.kron(np.kron(
    np.kron(np.eye(2), np.eye(2)), x), np.eye(2)), np.eye(2)), paulis))
s4 = list(map(lambda x: np.kron(np.kron(np.kron(
    np.kron(np.eye(2), np.eye(2)), np.eye(2)), x), np.eye(2)), paulis))
s5 = list(map(lambda x: np.kron(np.kron(np.kron(
    np.kron(np.eye(2), np.eye(2)), np.eye(2)), np.eye(2)), x), paulis))
# s2 = list(map(lambda x: np.kron(np.kron(np.eye(2), x), np.eye(2)), paulis))
# s3 = list(map(lambda x: np.kron(np.kron(np.eye(2), np.eye(2)), x), paulis))

H = np.zeros((32, 32), dtype=np.complex128)

for p1, p2, w in zip(s1, s2, weights):
    H -= p1 @ p2 * w
for p1, p2, w in zip(s2, s3, weights):
    H -= p1 @ p2 * w
for p1, p2, w in zip(s3, s4, weights):
    H -= p1 @ p2 * w
for p1, p2, w in zip(s4, s5, weights):
    H -= p1 @ p2 * w
for p1, p2, w in zip(s5, s1, weights):
    H -= p1 @ p2 * w

# print(H)

# H = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, 3]])
# L = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, 3]])
# F0 = np.ones((8, 8)) - np.eye(8)
# F0 = np.diag([0, 0, 0, 0, 0, 0, 0], k=1)
# F0 = F0 + np.conj(F0.T)
# print(F0)
eigs, eigvs = np.linalg.eigh(H)
print(eigs)
print(sum(eigs))
# F0 = eigvs @ F0 @ np.conj(eigvs.T)
# L = H

# Lhat = L - 1j * F1
# Hhat = H + 0.5 * (F1 @ L + np.conj(L).T @ F1)
# print(Hhat)
# print(Lhat)
# print(1j * H[0, 1] - 0.5 * np.conj(Lhat[0, 0]) * Lhat[0, 1])

# I = np.eye(8)
#
# A = (
#     -1j * (np.kron(I, H + F0) - np.kron((H + F0).T, I))
#     + np.kron(np.conj(L), L)
#     - 0.5 * (np.kron(I, np.conj(L.T) @ L) + np.kron(L.T @ np.conj(L), I))
# )

# print("Spectrum")
# print(np.linalg.eig(A))

# print(np.linalg.matrix_rank(A))
# print(A.shape)
# print(A @ np.reshape(np.eye(8), shape=64))
# print(A @ np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]))
