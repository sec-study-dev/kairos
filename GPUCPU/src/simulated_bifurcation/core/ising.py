

from typing import Optional, TypeVar, Union

import torch
from numpy import ndarray

from ..optimizer import SimulatedBifurcationEngine, SimulatedBifurcationOptimizer

SelfIsing = TypeVar("SelfIsing", bound="Ising")

class Ising:

    def __init__(
        self,
        J: Union[torch.Tensor, ndarray],
        h: Union[torch.Tensor, ndarray, None] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.dimension = J.shape[0]
        if isinstance(J, ndarray):
            J = torch.from_numpy(J)
        if isinstance(h, ndarray):
            h = torch.from_numpy(h)
        self.__init_from_tensor(J, h, dtype, device)
        self.computed_spins = None

    def __len__(self) -> int:
        return self.dimension

    def __neg__(self) -> SelfIsing:
        return self.__class__(-self.J, -self.h, self.dtype, self.device)

    def __init_from_tensor(
        self,
        J: torch.Tensor,
        h: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> None:
        null_vector = torch.zeros(self.dimension, dtype=dtype, device=device)
        self.J = J.to(device=device, dtype=dtype)
        if h is None:
            self.h = null_vector
            self.linear_term = False
        else:
            self.h = h.to(device=device, dtype=dtype)
            self.linear_term = not torch.equal(self.h, null_vector)

    def clip_vector_to_tensor(self) -> torch.Tensor:

        tensor = torch.zeros(
            (self.dimension + 1, self.dimension + 1),
            dtype=self.dtype,
            device=self.device,
        )
        tensor[: self.dimension, : self.dimension] = self.J
        tensor[: self.dimension, self.dimension] = -self.h
        tensor[self.dimension, : self.dimension] = -self.h
        return tensor

    @staticmethod
    def remove_diagonal_(tensor: torch.Tensor) -> None:

        torch.diagonal(tensor)[...] = 0

    @staticmethod
    def symmetrize(tensor: torch.Tensor) -> torch.Tensor:

        return (tensor + tensor.t()) / 2.0

    def as_simulated_bifurcation_tensor(self) -> torch.Tensor:

        tensor = self.symmetrize(self.J)
        self.remove_diagonal_(tensor)
        if self.linear_term:
            sb_tensor = self.clip_vector_to_tensor()
        else:
            sb_tensor = tensor
        return sb_tensor

    @property
    def dtype(self) -> torch.dtype:

        return self.J.dtype

    @property
    def device(self) -> torch.device:

        return self.J.device

    def minimize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> None:

        engine = SimulatedBifurcationEngine.get_engine(ballistic, heated)
        optimizer = SimulatedBifurcationOptimizer(
            agents,
            max_steps,
            timeout,
            engine,
            verbose,
            sampling_period,
            convergence_threshold,
        )
        tensor = self.as_simulated_bifurcation_tensor()
        spins = optimizer.run_integrator(tensor, use_window)
        if self.linear_term:
            self.computed_spins = spins[-1] * spins[:-1]
        else:
            self.computed_spins = spins
