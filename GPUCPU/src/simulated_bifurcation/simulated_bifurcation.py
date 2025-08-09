

from typing import Optional, Tuple, Union

import torch

from .core import PolynomialLike, QuadraticPolynomial

def build_model(
    *polynomial: PolynomialLike,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> QuadraticPolynomial:

    return QuadraticPolynomial(*polynomial, dtype=dtype, device=device)

def optimize(
    *polynomial: PolynomialLike,
    domain: str = "spin",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    agents: int = 128,
    max_steps: int = 10_000,
    best_only: bool = True,
    ballistic: bool = False,
    heated: bool = False,
    minimize: bool = True,
    verbose: bool = True,
    use_window: bool = True,
    sampling_period: int = 50,
    convergence_threshold: int = 50,
    timeout: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    model = build_model(
        *polynomial,
        dtype=dtype,
        device=device,
    )
    result, evaluation = model.optimize(
        domain=domain,
        agents=agents,
        max_steps=max_steps,
        best_only=best_only,
        ballistic=ballistic,
        heated=heated,
        minimize=minimize,
        verbose=verbose,
        use_window=use_window,
        sampling_period=sampling_period,
        convergence_threshold=convergence_threshold,
        timeout=timeout,
    )
    return result, evaluation

def minimize(
    *polynomial: PolynomialLike,
    domain: str = "spin",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    agents: int = 128,
    max_steps: int = 10_000,
    best_only: bool = True,
    ballistic: bool = False,
    heated: bool = False,
    verbose: bool = True,
    use_window: bool = True,
    sampling_period: int = 50,
    convergence_threshold: int = 50,
    timeout: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    return optimize(
        *polynomial,
        domain=domain,
        dtype=dtype,
        device=device,
        agents=agents,
        max_steps=max_steps,
        best_only=best_only,
        ballistic=ballistic,
        heated=heated,
        minimize=True,
        verbose=verbose,
        use_window=use_window,
        sampling_period=sampling_period,
        convergence_threshold=convergence_threshold,
        timeout=timeout,
    )

def maximize(
    *polynomial: PolynomialLike,
    domain: str = "spin",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    agents: int = 128,
    max_steps: int = 10_000,
    best_only: bool = True,
    ballistic: bool = False,
    heated: bool = False,
    verbose: bool = True,
    use_window: bool = True,
    sampling_period: int = 50,
    convergence_threshold: int = 50,
    timeout: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    return optimize(
        *polynomial,
        domain=domain,
        dtype=dtype,
        device=device,
        agents=agents,
        max_steps=max_steps,
        best_only=best_only,
        ballistic=ballistic,
        heated=heated,
        minimize=False,
        verbose=verbose,
        use_window=use_window,
        sampling_period=sampling_period,
        convergence_threshold=convergence_threshold,
        timeout=timeout,
    )
