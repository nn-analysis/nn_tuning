from typing import Union

import numpy as np

from .storage import TableSet, Table
from .stimulus_generator import StimulusGenerator


class InputManager:
    """
    Class that manages the input and the batches.
    Takes an input generator as input. The generator saves the input to a table.
    That table is then used to determine which batches are valid and to retrieve input.

    Args:
        table: (`Table` or `TableSet`)  The table the input generator stores it's data to.
        shape: (tuple) A tuple of the shape of the input so it can be transformed to that.
        stimulus_generator: (`StimulusGenerator`) An StimulusGenerator that can generate input.
    """

    def __init__(self, table: Union[Table, TableSet], shape: tuple, stimulus_generator: StimulusGenerator = None):
        self._table = table
        self._shape = shape
        if stimulus_generator is not None and (not self._table.initialised):
            stimulus_generator.generate(shape)

    @staticmethod
    def __prod(val):
        """
        The product of all iterable variables together

        Examples
        ----------
        >>> __prod((1,2,3,4,5))
        120
        >>> __prod((8, 12, 5))
        480

        Args:
            val: tuple

        Returns:
            The product as a float
        """
        res = 1
        for ele in val:
            res *= ele
        return res

    def valid(self, batch: int, batch_size: int) -> bool:
        """
        Determines if the batch is valid for the input table

        Examples
        ------------
        >>> valid(1, 100)
        True
        >>> valid(10, 100)
        False

        Args:
            batch: Integer of the batch that needs to be tested for validity
            batch_size: The size of the batches.

        Returns:
            boolean depicting the validity of the batch.
        """
        return batch * batch_size < self._table.nrows

    def get(self, batch: int, batch_size: int) -> np.ndarray:
        """
        Function to get the input for a specific batch.

        Examples
        ----------
        >>> get(0, 2)
        Array([
            [1,1,1,1,1,1,0,0,0,0,0,0,0,...],
            [0,1,1,1,1,1,1,0,0,0,0,0,0,...]
        ])
        >>> get(0, 4)
        Array([
            [1,1,1,1,1,1,0,0,0,0,0,0,0,...],
            [0,1,1,1,1,1,1,0,0,0,0,0,0,...],
            [0,1,1,1,1,1,1,0,0,0,0,0,0,...],
            [0,1,1,1,1,1,1,0,0,0,0,0,0,...]
        ])

        Args:
            batch: Integer of the batch.
            batch_size: The size of the batch.

        Returns:
            np.ndarray containing the input
        """
        start = batch * batch_size
        end = batch * batch_size + batch_size
        if end > self._table.nrows:
            end = self._table.nrows
        return self._table[start:end].reshape(-1, *self._shape)
