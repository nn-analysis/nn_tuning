from abc import ABC
import numpy as np


class InputManager(ABC):

    def valid(self, batch: int, batch_size: int) -> bool:
        raise NotImplementedError

    def get(self, batch: int, batch_size: int, verbose: bool) -> np.array:
        raise NotImplementedError
