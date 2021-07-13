from typing import Union

import numpy as np
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    no_plotting = False
except ImportError:
    no_plotting = True
    plt = None
import nn_tuning.plot as plot
from ..storage import StorageManager, Table, TableSet
from .two_d_stimulus_generator import TwoDStimulusGenerator


class PRFStimulusGenerator(TwoDStimulusGenerator):
    """
    Class that generates pRF stimuli for visual field position in a moving bar.

    Args:
        stride: The stride of the moving bar.
        table: The name of the table to save the stimuli to.
        storage_manager: The StorageManager to use to save the results.
        block_size (optional, default=5): The size of the blocks in the stimulus.
        verbose (optional, default=False): Whether the class should print its progress to the console.
    """

    @property
    def stim_y(self):
        stim_y = []
        for x in range(1, self.shape[0] + 1):
            for y in range(1, self.shape[1] + 1):
                stim_y.append(y)
        return np.array(stim_y)

    @property
    def stim_x(self):
        stim_x = []
        for x in range(1, self.shape[0] + 1):
            for y in range(1, self.shape[1] + 1):
                stim_x.append(x)
        return np.array(stim_x)

    def __init__(self, stride: int, shape: (int, int), table: str, storage_manager: StorageManager, block_size: int = 5,
                 verbose: bool = False):
        self.__stride = stride
        self.__table = table
        self.__storage_manager = storage_manager
        self.__verbose = verbose
        self.shape = shape
        self.__block_size = block_size

    def _get_2d(self, shape: (int, int), index: int) -> np.ndarray:
        """Generates the 2d stimulus to be appended with other dimensions to a complete stimulus.

        Args:
            shape: (int, int) The shape of the 2d stimulus to generate.
            index: (int) The index of the stimulus. The index allows the function to differentiate which variation to generate in a generalisable way.

        Returns:
            (np.ndarray) The generated stimulus as a 2d image.
        """
        size_x = shape[1]
        size_y = shape[0]
        block_size = self.__block_size
        checkerboard = np.zeros(shape)
        mask = np.zeros(shape)

        for j in range(5):
            for i in range(5):
                checkerboard[j::block_size * 2, i::block_size * 2] = 1
                checkerboard[j + block_size::block_size * 2, i + block_size::block_size * 2] = 1

        if index < size_x + 1:
            usable_index = index
            if usable_index < block_size:
                start = 0
            else:
                start = usable_index - block_size
            if usable_index > size_x - block_size:
                end = size_x
            else:
                end = usable_index + block_size
            mask[:, start:end] = 1
            result = checkerboard * mask
            result[:, 0:start] = 0.5
            result[:, end:size_x] = 0.5
        else:
            usable_index = index - size_x - 1
            if usable_index < block_size:
                start = 0
            else:
                start = usable_index - block_size
            if usable_index > size_y - block_size:
                end = size_y
            else:
                end = usable_index + block_size
            mask[start:end] = 1
            result = checkerboard * mask
            result[0:start] = 0.5
            result[end:size_y] = 0.5
        result = result.reshape(-1)
        return result

    def generate(self, shape: tuple) -> Union[Table, TableSet]:
        """
        Generates all input and saves the input to a table

        Args:
            shape: (tuple) The expected shape of the input

        Returns:
            `Table` or `TableSet` containing the stimuli
        """
        tbl = None
        size_x = shape[-1]
        size_y = shape[-2]
        for i in tqdm(range(0, size_x + size_y + 2, self.__stride), leave=False, disable=(not self.__verbose)):
            tbl = self.__storage_manager.save_result_table_set((self._generate_row(shape, i)[np.newaxis, ...],),
                                                               self.__table, {self.__table: self.__table},
                                                               append_rows=True)
        return tbl

    @property
    def stimulus_description(self) -> np.ndarray:
        """
        Generates the stimulus description for the pRF data to be used by the `FittingManager`.
        The shape indicates the shape of the 2d images.

        Returns:
            np.ndarray containing the stimulus variable to be used by the FittingManager.
        """
        results = np.zeros((self.shape[0] + self.shape[1] + 2, *self.shape))
        size_x = self.shape[1]
        size_y = self.shape[0]
        for i in range(0, size_x + 1):
            if i < 5:
                start = 0
            else:
                start = i - 5
            if i > size_x + 1 - 5:
                end = size_x + 1
            else:
                end = i + 5
            results[i, :, start:end] = 1
        for i in range(0, size_y + 1):
            if i < 5:
                start = 0
            else:
                start = i - 5
            if i > size_y + 1 - 5:
                end = size_x + 1
            else:
                end = i + 5
            results[i + size_x + 1, start:end, :] = 1
        return results.reshape(results.shape[0], -1)

    def plot_image(self, shape: (int, int), index: int, title: str):
        """
        Generates and plots a stimulus with a certain shape at a certain index.
        The title will either be displayed in the plot (when the plot is shown), or used as a filename (when the plot is saved).
        This function uses the Plot class to save or store plots. To save the plot as an image set `Plot.save_fig` to `True`.

        Examples
        ----------
        >>> PRFStimulusGenerator().plot_image((128, 160), 10, 'Title')
        Plots an image with a size 160x128 at index 10 with title 'Title'.

        Args:
            shape: (int, int) Shape of the image
            index: (int) Index of the stimulus
            title: (str) Title of the plot or the filename of the plot
        """
        if no_plotting:
            raise ImportError('Plotting requires the matplotlib package. Please install the package and try again.')
        result = self._get_2d(shape, index).reshape(shape)
        plt.imshow(result, cmap='gray', vmin=0, vmax=1, origin='lower')
        if plot.save_fig:
            plot.title = title
        else:
            plt.title(title)
        plot.show(plt)
        plot.title = None
