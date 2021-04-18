import numpy as np
from code_analysis.storage_manager import StorageManager
from code_analysis.input_generator import InputGenerator
from code_analysis.input_manager import InputManager


class NDInputManager(InputManager):

    def __init__(self, table: str, shape: tuple, storage_manager: StorageManager,
                 input_generator: InputGenerator = None):
        self._table = storage_manager.open_table(table)
        self._shape = shape
        if input_generator is not None and (not self._table.initialised):
            input_generator.generate(shape)

    def valid(self, batch: int, batch_size: int) -> bool:
        return batch * batch_size < self._table.nrows

    def get(self, batch: int, batch_size: int) -> np.array:
        start = batch * batch_size
        end = batch * batch_size + batch_size
        if end > self._table.nrows:
            end = self._table.nrows
        return self._table[start:end].reshape(-1, *self._shape)
