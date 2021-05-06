from abc import ABC

import numpy as np

from code_analysis import InputGenerator


class TwoDInputGenerator(InputGenerator, ABC):

    def _get_2d(self, shape: (int, int), index: int) -> np.array:
        raise NotImplementedError

    def _generate_row(self, shape: tuple, index: int):
        if len(shape) < 2:
            raise ValueError(f'expected len(shape) > 2, got {len(shape)}')
        length = len(shape)
        shapes = list(range(0, length-2))
        shapes.reverse()
        image = self._get_2d((shape[-2], shape[-1]), index)
        new_image = image
        for i in shapes:
            tmp_arr = []
            for j in range(0, shape[i]):
                tmp_arr.append(new_image)
            new_image = np.array(tmp_arr)
        return new_image.reshape(-1)
