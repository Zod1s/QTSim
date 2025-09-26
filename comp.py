import numpy as np

paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1j], [1j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

# H = -pauliz - 0.1 * pauliy
# L = -pauliz + 0.1 * paulix
# F0 = -0.1 * pauliy
H = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, 3]])
L = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, 3]])
F0 = 0.1 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Lhat = L - 1j * F1
# Hhat = H + 0.5 * (F1 @ L + np.conj(L).T @ F1)
# print(Hhat)
# print(Lhat)
# print(1j * H[0, 1] - 0.5 * np.conj(Lhat[0, 0]) * Lhat[0, 1])

I = np.eye(3)

A = -1j * (np.kron(I, H + F0) - np.kron((H + F0).T, I)) + \
    np.kron(np.conj(L), L) - 0.5 * \
    (np.kron(I, np.conj(L.T) @ L) + np.kron(L.T @ np.conj(L), I))

print(A)
print(A.real)
print(A.imag)

print(np.linalg.matrix_rank(A))

print(A @ np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]))


# print(A)
# eigs, eigvs = np.linalg.eig(A)
# print(eigs)
# print(eigvs)
