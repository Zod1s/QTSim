%% Initialisation
clear all
close all
clc

%% Gell-Mann matrices
gellmann(:, :, 1) = [0 1 0; 1 0 0; 0 0 0] / sqrt(2);
gellmann(:, :, 2) = [0 -1j 0; 1j 0 0; 0 0 0] / sqrt(2);
gellmann(:, :, 3) = [1 0 0; 0 -1 0; 0 0 0] / sqrt(2);
gellmann(:, :, 4) = [0 0 1; 0 0 0; 1 0 0] / sqrt(2);
gellmann(:, :, 5) = [0 0 -1j; 0 0 0; 1j 0 0] / sqrt(2);
gellmann(:, :, 6) = [0 0 0; 0 0 1; 0 1 0] / sqrt(2);
gellmann(:, :, 7) = [0 0 0; 0 0 -1j; 0 1j 0] / sqrt(2);
gellmann(:, :, 8) = [1 / sqrt(3) 0 0; 0 1 / sqrt(3) 0; 0 0 -2 / sqrt(3)] / sqrt(2);

%% Hamiltonian and measurement operator
H = diag([-1, 2, 3]);
L = diag([-1, 2, 3]);
F0 = [0 1 0; 1 0 1; 0 1 0];
H = H + F0;

lindbladian = @(x) -1j * (H * x - x * H) + L * x * L' - 0.5 * (L' * L * x + x * L' * L);

%% Coherence matrix
Al = zeros(8, 8);

for i = 1:8
    for j = 1:8
        Al(i, j) = trace(gellmann(:, :, i)' * lindbladian(gellmann(:, :, j)));
    end
end

Q = eye(8);
P = lyap(Al', Q);
eig(P)