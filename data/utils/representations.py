import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numba
import numpy as np
import torch
import torch as th
from einops import rearrange, reduce


class RepresentationBase(ABC):
    @abstractmethod
    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        ...

    @abstractmethod
    def get_shape(self) -> Tuple[int, int, int]:
        ...

    @staticmethod
    @abstractmethod
    def get_numpy_dtype() -> np.dtype:
        ...

    @staticmethod
    @abstractmethod
    def get_torch_dtype() -> th.dtype:
        ...

    @property
    def dtype(self) -> th.dtype:
        return self.get_torch_dtype()

    @staticmethod
    def _is_int_tensor(tensor: th.Tensor) -> bool:
        return not th.is_floating_point(tensor) and not th.is_complex(tensor)


class StackedHistogram(RepresentationBase):
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int] = None, fast_mode: bool = True):
        """
        In case of fast-mode == True: use uint8 to construct the representation, but could lead to overflow.
        In case of fast-mode == False: use int16 to construct the representation, and convert to uint8 after clipping.

        Note: Overflow should not be a big problem because it happens only for hot pixels. In case of overflow,
        the value will just start accumulating from 0 again.
        """
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is None:
            self.count_cutoff = 255
        else:
            assert count_cutoff >= 1
            self.count_cutoff = min(count_cutoff, 255)
        self.fast_mode = fast_mode
        self.channels = 2

    @staticmethod
    def get_numpy_dtype() -> np.dtype:
        return np.dtype('uint8')

    @staticmethod
    def get_torch_dtype() -> th.dtype:
        return th.uint8

    def merge_channel_and_bins(self, representation: th.Tensor):
        assert representation.dim() == 4
        return th.reshape(representation, (-1, self.height, self.width))

    def get_shape(self) -> Tuple[int, int, int]:
        return 2 * self.bins, self.height, self.width

    def construct(self, x: th.Tensor, y: th.Tensor, pol: th.Tensor, time: th.Tensor) -> th.Tensor:
        device = x.device
        assert y.device == pol.device == time.device == device
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        dtype = th.uint8 if self.fast_mode else th.int16

        representation = th.zeros((self.channels, self.bins, self.height, self.width),
                                  dtype=dtype, device=device, requires_grad=False)

        if x.numel() == 0:
            assert y.numel() == 0
            assert pol.numel() == 0
            assert time.numel() == 0
            return self.merge_channel_and_bins(representation.to(th.uint8))
        assert x.numel() == y.numel() == pol.numel() == time.numel()

        assert pol.min() >= 0
        assert pol.max() <= 1

        bn, ch, ht, wd = self.bins, self.channels, self.height, self.width

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        assert t1_int >= t0_int
        t_norm = time - t0_int
        t_norm = t_norm / max((t1_int - t0_int), 1)
        t_norm = t_norm * bn
        t_idx = t_norm.floor()
        t_idx = th.clamp(t_idx, max=bn - 1)

        indices = (x.long() +
                   wd * y.long() +
                   ht * wd * t_idx.long() +
                   bn * ht * wd * pol.long())
        indices = torch.clamp(indices, max=bn * ht * wd * ch - 1)
        values = th.ones_like(indices, dtype=dtype, device=device)
        representation.put_(indices, values, accumulate=True)
        representation = th.clamp(representation, min=0, max=self.count_cutoff)
        if not self.fast_mode:
            representation = representation.to(th.uint8)

        return representation
        # return self.merge_channel_and_bins(representation)


def get_file_list(file_path: str) -> list:
    return os.listdir(file_path)


def merge_channel_and_bins(representation: th.Tensor):
    assert representation.dim() == 4
    return th.reshape(representation, (-1, 200, 300))
