import numpy as np

paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1j], [1j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

H = -pauliz - 0.1 * pauliy
L = -pauliz + 0.1 * paulix
F1 = -0.1 * pauliy
print(np.linalg.eigh(H))


Lhat = L - 1j * F1
Hhat = H + 0.5 * (F1 @ L + np.conj(L).T @ F1)
# print(Hhat)
# print(Lhat)
# print(1j * H[0, 1] - 0.5 * np.conj(Lhat[0, 0]) * Lhat[0, 1])

I = np.eye(2)
LhatdLhat = np.conj(Lhat).T @ Lhat

A = (
    -1j * (np.kron(I, Hhat) - np.kron(Hhat.T, I))
    + np.kron(np.conj(Lhat), Lhat)
    - 0.5 * (np.kron(I, LhatdLhat) + np.kron(LhatdLhat.T, I))
)

# print(A)
# eigs, eigvs = np.linalg.eig(A)
# print(eigs)
# print(eigvs)
