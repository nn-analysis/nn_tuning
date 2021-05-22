from abc import ABC

import numpy as np

from .input_generator import InputGenerator


class TwoDInputGenerator(InputGenerator, ABC):
    """
    Abstract class with useful functions for InputGenerators that makes it so you do not need to worry about the
    reshaping of rows in an individual stimulus and can focus on the 2d input image.

    To implement this class, call the _generate_row function from the generate function and implement the _get_2d function.
    The _get_2d function handles the creation of 2d stimuli while the _generate_row function handles creating the right shapes around the stimuli.
    """

    def _get_2d(self, shape: (int, int), index: int) -> np.array:
        """Generates the 2d stimulus to be appended with other dimensions to a complete stimulus.

        Args:
            shape (:obj:`(int, int)`) : The shape of the 2d stimulus to generate.
            index (int) : The index of the stimulus. The index allows the function to differentiate which variation to generate in a generalisable way.

        Returns:
            object (:obj:`np.array`) : The generated stimulus as a 2d image.
        """
        raise NotImplementedError

    def _generate_row(self, shape: tuple, index: int):
        """
        Generates a single stimuli. This specific function provides the reshaping and uses `_get_2d` to generate the actual stimuli images.

        Args:
            shape: Total shape of the stimuli including colour and time channels
            index: The index of the stimulus. This allows `_get_2d` to know which stimulus to generate.

        Returns:
            np.array containing a single stimulus.
        """
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
