import abc
from abc import ABC

import numpy as np

from ..storage import Table, TableSet
from typing import Union


class StimulusGenerator(ABC):
    """
    Abstract class for the StimulusGenerators.
    """

    @abc.abstractmethod
    def generate(self, shape: tuple) -> Union[Table, TableSet]:
        """
        Generates all input and saves the input to a table.

        Usage
        ------
        >>> StimulusGenerator().generate((128,160))

        Args:
            shape: The expected shape of the input

        Returns:
            Table or TableSet containing the stimuli
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stimulus_description(self) -> np.ndarray:
        """
        Generates the stimulus description for use in the `FittingManager`

        Returns:
            np.ndarray containing the stimulus variable to be used by the FittingManager.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stim_x(self):
        pass

    @property
    @abc.abstractmethod
    def stim_y(self):
        pass
