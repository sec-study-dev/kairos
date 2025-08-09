import torch
import simulated_bifurcation as sb

matrix = torch.tensor(
    [
        [1, 1, 2],
        [0, -1, -2],
        [-2, 0, 2],
    ],
    dtype=torch.float32,
)
vector = torch.tensor([-1, 0, 2], dtype=torch.float32)
constant = 2.0

polynomial = sb.build_model(matrix, vector, constant)

spin_value, spin_vector = sb.minimize(matrix, vector, constant, domain='spin')

binary_value, binary_vector = sb.minimize(matrix, vector, constant, domain='binary')

int_value, int_vector = sb.minimize(matrix, vector, constant, domain='int3')
print(int_value)
print(int_vector)

