import numpy as np

from code_analysis import StorageManager, Table, InputGenerator, plt, Plot


class PRFInputGenerator(InputGenerator):

    def __init__(self, stride: int, table: str, storage_manager: StorageManager, block_size: int = 5, verbose: bool = False):
        self.__stride = stride
        self.__table = table
        self.__storage_manager = storage_manager
        self.__verbose = verbose
        self.__block_size = block_size

    def _get_2d(self, shape: (int, int), index: int) -> np.array:
        size_x = shape[1]
        size_y = shape[0]
        block_size = self.__block_size
        checkerboard = np.zeros(shape)
        mask = np.zeros(shape)

        for j in range(5):
            for i in range(5):
                checkerboard[j::block_size*2, i::block_size*2] = 1
                checkerboard[j+block_size::block_size*2, i+block_size::block_size*2] = 1

        if index < size_x+1:
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

    def generate(self, shape: tuple):
        tbl = None
        col_index = Table.shape_to_indices(shape)
        size_x = shape[-1]
        size_y = shape[-2]
        for i in range(0, size_x + size_y + 2, self.__stride):
            if self.__verbose:
                print(f'{int(i / (size_x + size_y + 2) * 100)}%', end='')
            tbl = self.__storage_manager.save_results(self.__table, self._generate_row(shape, i)[np.newaxis, ...],
                                                      [i], col_index)
        if self.__verbose:
            print()
        return tbl

    @staticmethod
    def get_stimulus(shape: (int, int)):
        results = np.zeros((shape[0] + shape[1] + 2, *shape))
        size_x = shape[1]
        size_y = shape[0]
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
            results[i+size_x+1, start:end, :] = 1
        return results.reshape(results.shape[0], -1)

    def plot_image(self, shape: (int, int), index: int, title: str):
        result = self._get_2d(shape, index).reshape(shape)
        plt.imshow(result, cmap='gray', vmin=0, vmax=1, origin='lower')
        if Plot.save_fig:
            Plot.title = title
        else:
            plt.title(title)
        Plot.show(plt)
        Plot.title = None
