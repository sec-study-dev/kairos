

import re
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..polynomial import Polynomial, PolynomialLike
from .ising import Ising

INTEGER_REGEX = re.compile("^int[1-9][0-9]*$")
DOMAIN_ERROR = ValueError(
    f'Input type must be one of "spin" or "binary", or be a string starting'
    f'with "int" and be followed by a positive integer.\n'
    f"More formally, it should match the following regular expression.\n"
    f"{INTEGER_REGEX}\n"
    f'Examples: "int7", "int42", ...'
)

class QuadraticPolynomialError(ValueError):
    def __init__(self, degree: int) -> None:
        super().__init__(f"Expected a degree 2 polynomial, got {degree}.")

class QuadraticPolynomial(Polynomial):

    def __init__(
        self,
        *polynomial_like: PolynomialLike,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(*polynomial_like, dtype=dtype, device=device)
        self.sb_result = None
        if self.degree != 2:
            raise QuadraticPolynomialError(self.degree)

    def __call__(self, value: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            try:
                value = torch.tensor(value, dtype=self.dtype, device=self.device)
            except Exception as err:
                raise TypeError("Input value cannot be cast to Tensor.") from err

        if value.shape[-1] != self.n_variables:
            raise ValueError(
                f"Size of the input along the last axis should be "
                f"{self.n_variables}, it is {value.shape[-1]}."
            )

        quadratic_term = torch.nn.functional.bilinear(
            value,
            value,
            torch.unsqueeze(self[2], 0),
        )
        affine_term = value @ self[1] + self[0]
        evaluation = torch.squeeze(quadratic_term, -1) + affine_term
        return evaluation

    def to_ising(self, domain: str) -> Ising:

        if domain == "spin":
            return Ising(-2 * self[2], self[1], self.dtype, self.device)
        if domain == "binary":
            symmetrical_matrix = Ising.symmetrize(self[2])
            J = -0.5 * symmetrical_matrix
            h = 0.5 * self[1] + 0.5 * symmetrical_matrix @ torch.ones(
                self.n_variables, dtype=self.dtype, device=self.device
            )
            return Ising(J, h, self.dtype, self.device)
        if INTEGER_REGEX.match(domain) is None:
            raise DOMAIN_ERROR
        number_of_bits = int(domain[3:])
        symmetrical_matrix = Ising.symmetrize(self[2])
        integer_to_binary_matrix = QuadraticPolynomial.__integer_to_binary_matrix(
            self.n_variables, number_of_bits, device=self.device
        )
        J = (
            -0.5
            * integer_to_binary_matrix
            @ symmetrical_matrix
            @ integer_to_binary_matrix.t()
        )
        h = 0.5 * integer_to_binary_matrix @ self[
            1
        ] + 0.5 * integer_to_binary_matrix @ self[
            2
        ] @ integer_to_binary_matrix.t() @ torch.ones(
            (self.n_variables * number_of_bits),
            dtype=self.dtype,
            device=self.device,
        )
        return Ising(J, h, self.dtype, self.device)

    def convert_spins(self, ising: Ising, domain: str) -> Optional[torch.Tensor]:

        if ising.computed_spins is None:
            return None
        if domain == "spin":
            return ising.computed_spins
        if domain == "binary":
            return (ising.computed_spins + 1) / 2
        if INTEGER_REGEX.match(domain) is None:
            raise DOMAIN_ERROR
        number_of_bits = int(domain[3:])
        integer_to_binary_matrix = QuadraticPolynomial.__integer_to_binary_matrix(
            self.n_variables, number_of_bits, device=self.device
        )
        return 0.5 * integer_to_binary_matrix.t() @ (ising.computed_spins + 1)

    def optimize(
        self,
        domain: str,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        minimize: bool = True,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if minimize:
            ising_equivalent = self.to_ising(domain)
        else:
            ising_equivalent = -self.to_ising(domain)
        ising_equivalent.minimize(
            agents,
            max_steps,
            ballistic,
            heated,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )
        self.sb_result = self.convert_spins(ising_equivalent, domain)
        result = self.sb_result.t().to(dtype=self.dtype)
        evaluation = self(result)
        if best_only:
            i_best = torch.argmin(evaluation) if minimize else torch.argmax(evaluation)
            result = result[i_best]
            evaluation = evaluation[i_best]
        return result, evaluation

    def minimize(
        self,
        domain: str,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.optimize(
            domain,
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            True,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )

    def maximize(
        self,
        domain: str,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.optimize(
            domain,
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            False,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )

    @staticmethod
    def __integer_to_binary_matrix(
        dimension: int, number_of_bits: int, device: Union[str, torch.device]
    ) -> torch.Tensor:

        matrix = torch.zeros((dimension * number_of_bits, dimension), device=device)
        for row in range(dimension):
            for col in range(number_of_bits):
                matrix[row * number_of_bits + col][row] = 2.0**col
        return matrix
