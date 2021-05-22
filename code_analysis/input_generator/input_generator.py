import abc
from abc import ABC
from ..storage import Table, TableSet
from typing import Union


class InputGenerator(ABC):
    """
    Abstract class for the InputGenerators.
    """

    @abc.abstractmethod
    def generate(self, shape: tuple) -> Union[Table, TableSet]:
        """
        Generates all input and saves the input to a table

        Args:
            shape: The expected shape of the input

        Returns:
            Table or TableSet containing the stimuli
        """
        raise NotImplementedError
