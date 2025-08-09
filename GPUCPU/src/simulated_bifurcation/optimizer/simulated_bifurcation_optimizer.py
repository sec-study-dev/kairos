import logging
import warnings
from time import time
from typing import Optional, Tuple

import torch
from numpy import minimum
from tqdm import tqdm

from .environment import ENVIRONMENT
from .simulated_bifurcation_engine import SimulatedBifurcationEngine
from .stop_window import StopWindow
from .symplectic_integrator import SymplecticIntegrator

LOGGER = logging.getLogger("simulated_bifurcation_optimizer")
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.set_name(logging.WARN)
CONSOLE_HANDLER.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
LOGGER.addHandler(CONSOLE_HANDLER)

class ConvergenceWarning(Warning):
    def __str__(self) -> str:
        return "No agent has converged. Returned signs of final positions instead."

class SimulatedBifurcationOptimizer:

    def __init__(
        self,
        agents: int,
        max_steps: Optional[int],
        timeout: Optional[float],
        engine: SimulatedBifurcationEngine,
        verbose: bool,
        sampling_period: int,
        convergence_threshold: int,
    ) -> None:

        self.engine = engine
        self.window = None
        self.symplectic_integrator = None
        self.heat_coefficient = ENVIRONMENT.heat_coefficient
        self.heated = engine.heated
        self.verbose = verbose
        self.start_time = None
        self.simulation_time = None

        self.time_step = ENVIRONMENT.time_step
        self.agents = agents
        self.pressure_slope = ENVIRONMENT.pressure_slope

        self.convergence_threshold = convergence_threshold
        self.sampling_period = sampling_period
        self.max_steps = max_steps if max_steps is not None else float("inf")
        self.timeout = timeout if timeout is not None else float("inf")

    def __reset(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.__init_progress_bars()
        self.__init_symplectic_integrator(matrix)
        self.__init_window(matrix, use_window)
        self.__init_quadratic_scale_parameter(matrix)
        self.run = True
        self.step = 0
        self.start_time = None
        self.simulation_time = 0

    def __init_progress_bars(self) -> None:
        self.iterations_progress = tqdm(
            total=self.max_steps,
            desc="🔁 Iterations       ",
            disable=not self.verbose or self.max_steps == float("inf"),
            smoothing=0.1,
            mininterval=0.5,
            unit=" steps",
        )
        self.time_progress = tqdm(
            total=self.timeout,
            desc="⏳ Simulation time  ",
            disable=not self.verbose or self.timeout == float("inf"),
            smoothing=0.1,
            mininterval=0.5,
            bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} seconds",
        )

    def __init_quadratic_scale_parameter(self, matrix: torch.Tensor):
        self.quadratic_scale_parameter = (
            0.5 * (matrix.shape[0] - 1) ** 0.5 / (torch.sqrt(torch.sum(matrix**2)))
        )

    def __init_window(self, matrix: torch.Tensor, use_window: bool) -> None:
        self.window = StopWindow(
            matrix,
            self.agents,
            self.convergence_threshold,
            matrix.dtype,
            matrix.device,
            (self.verbose and use_window),
        )

    def __init_symplectic_integrator(self, matrix: torch.Tensor) -> None:
        self.symplectic_integrator = SymplecticIntegrator(
            (matrix.shape[0], self.agents),
            self.engine.activation_function,
            matrix.dtype,
            matrix.device,
        )

    def __step_update(self) -> None:
        self.step += 1
        self.iterations_progress.update()

    def __check_stop(self, use_window: bool) -> None:
        if use_window and self.__do_sampling:
            self.run = self.window.must_continue()
            if not self.run:
                LOGGER.info("Optimizer stopped. Reason: all agents converged.")
                return
        if self.step >= self.max_steps:
            self.run = False
            LOGGER.info(
                "Optimizer stopped. Reason: maximum number of iterations reached."
            )
            return
        previous_time = self.simulation_time
        self.simulation_time = time() - self.start_time
        time_update = min(
            self.simulation_time - previous_time, self.timeout - previous_time
        )
        self.time_progress.update(time_update)
        if self.simulation_time > self.timeout:
            self.run = False
            LOGGER.info("Optimizer stopped. Reason: computation timeout reached.")
            return

    @property
    def __do_sampling(self) -> bool:
        return self.step % self.sampling_period == 0

    def __close_progress_bars(self):
        self.iterations_progress.close()
        self.time_progress.close()
        self.window.progress.close()

    def __symplectic_update(
        self,
        matrix: torch.Tensor,
        use_window: bool,
    ) -> torch.Tensor:
        self.start_time = time()
        while self.run:
            if self.heated:
                momentum_copy = self.symplectic_integrator.momentum.clone()

            (
                momentum_coefficient,
                position_coefficient,
                quadratic_coefficient,
            ) = self.__compute_symplectic_coefficients()
            self.symplectic_integrator.step(
                momentum_coefficient,
                position_coefficient,
                quadratic_coefficient,
                matrix,
            )

            if self.heated:
                self.__heat(momentum_copy)

            self.__step_update()
            if use_window and self.__do_sampling:
                sampled_spins = self.symplectic_integrator.sample_spins()
                self.window.update(sampled_spins)

            self.__check_stop(use_window)

        sampled_spins = self.symplectic_integrator.sample_spins()
        return sampled_spins

    def __heat(self, momentum_copy: torch.Tensor) -> None:
        torch.add(
            self.symplectic_integrator.momentum,
            momentum_copy,
            alpha=self.time_step * self.heat_coefficient,
            out=self.symplectic_integrator.momentum,
        )

    def __compute_symplectic_coefficients(self) -> Tuple[float, float, float]:
        pressure = self.__pressure
        position_coefficient = self.time_step
        momentum_coefficient = self.time_step * (pressure - 1.0)
        quadratic_coefficient = self.time_step * self.quadratic_scale_parameter
        return momentum_coefficient, position_coefficient, quadratic_coefficient

    @property
    def __pressure(self):
        return minimum(self.time_step * self.step * self.pressure_slope, 1.0)

    def run_integrator(self, matrix: torch.Tensor, use_window: bool) -> torch.Tensor:

        if (
            self.max_steps == float("inf")
            and self.timeout == float("inf")
            and not use_window
        ):
            raise ValueError("No stopping criterion provided.")
        self.__reset(matrix, use_window)
        spins = self.__symplectic_update(matrix, use_window)
        self.__close_progress_bars()
        return self.get_final_spins(spins, use_window)

    def get_final_spins(self, spins: torch.Tensor, use_window: bool) -> torch.Tensor:

        if use_window:
            if not self.window.has_bifurcated_spins():
                warnings.warn(ConvergenceWarning(), stacklevel=2)
            return self.window.get_bifurcated_spins(spins)
        else:
            return spins
