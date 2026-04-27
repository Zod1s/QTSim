import numpy as np

paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1j], [1j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

paulis = [paulix, pauliy, pauliz]
weights = [1, 1, 2]

s1 = list(map(lambda x: np.kron(np.kron(x, np.eye(2)), np.eye(2)), paulis))
s2 = list(map(lambda x: np.kron(np.kron(np.eye(2), x), np.eye(2)), paulis))
s3 = list(map(lambda x: np.kron(np.kron(np.eye(2), np.eye(2)), x), paulis))

H = np.zeros((8, 8), dtype=np.complex128)

for (p1, p2, w) in zip(s1, s2, weights):
    H -= p1 @ p2 * w
for (p1, p2, w) in zip(s2, s3, weights):
    H -= p1 @ p2 * w
for (p1, p2, w) in zip(s3, s1, weights):
    H -= p1 @ p2 * w

F0 = np.diag([1, 1, 1, 1, 1, 1, 1], k=1)
F0 = F0 + np.conj(F0.T)
F0 = F0 * 4
print(F0)
eigs, eigvs = np.linalg.eigh(H)
print(eigs)
F0 = eigvs @ F0 @ np.conj(eigvs.T)
L = H

I = np.eye(8)

A = -1j * (np.kron(I, H + F0) - np.kron((H + F0).T, I)) + \
    np.kron(np.conj(L), L) - 0.5 * \
    (np.kron(I, np.conj(L.T) @ L) + np.kron(L.T @ np.conj(L), I))

print(np.linalg.matrix_rank(A))
print(A.shape)
print(A @ np.reshape(np.eye(8), shape=64))
