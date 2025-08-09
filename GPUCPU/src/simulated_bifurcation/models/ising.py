from typing import Optional, Union

import numpy as np
import torch

from .abc_model import ABCModel

class Ising(ABCModel):

    domain = "spin"

    def __init__(
        self,
        J: Union[torch.Tensor, np.ndarray],
        h: Union[torch.Tensor, np.ndarray, None] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(-0.5 * J, h, dtype=dtype, device=device)
        self.J = J
        self.h = h
