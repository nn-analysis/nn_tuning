import numpy as np

from code_analysis import StorageManager, Table, InputGenerator


class PositionalInputGenerator(InputGenerator):

    def __init__(self, amount_of_pixels: int, stride: int, table: str, size: int,
                 storage_manager: StorageManager, verbose: bool = False):
        self.__amount_of_pixels = amount_of_pixels
        self.__stride = stride
        self.__table = table
        self.__size = size
        self.__storage_manager = storage_manager
        self.__verbose = verbose

    def _get_2d(self, shape: (int, int), index: int) -> np.array:
        result = np.zeros(shape).reshape(-1)
        result[index:index+self.__amount_of_pixels] = 1
        return result

    def generate(self, shape: tuple):
        tbl = None
        col_index = Table.shape_to_indices(shape)
        for i in range(0, self.__size, self.__stride):
            if self.__verbose:
                print(f'{int(i / self.__size * 100)}%', end='')
            tbl = self.__storage_manager.save_results(self.__table, self._generate_row(shape, i)[np.newaxis, ...],
                                                      [i], col_index)
        if self.__verbose:
            print()
        return tbl
