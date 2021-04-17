import numpy as np

from .input_manager import InputManager
from .network import Network
from .storage_manager import StorageManager


class OutputManager:

    def __init__(self, network: Network, storage_manager: StorageManager, input_manager: InputManager):
        self.network = network
        self.storage_manager = storage_manager
        self.input_manager = input_manager

    def run(self, table: str, batch_size: int, override: bool = True, resume: bool = False, verbose: bool = False):
        """
        Function runs a batch through a network
        @param verbose: outputs the current batch and total percentage done
        @param table: Table to save data to
        @param batch_size: Size of the batches to input into the network
        @param override: bool determines whether the table gets extended or overridden (default=False)
        @param resume: resume at last batch on failure
        """
        batch = 0
        if override and not resume:
            self.storage_manager.remove_table(table)
        if resume:
            tbl = self.storage_manager.open_table(table)
            if tbl.initialised:
                batch = int(np.ceil(tbl.nrows/batch_size))
            else:
                batch = 0
        while self.input_manager.valid(batch, batch_size):
            self.network.current_batch = batch
            network_input = self.input_manager.get(batch, batch_size, verbose)
            network_output = self.network.run(network_input)
            row, column = self.network.get_indexes()
            self.storage_manager.save_results(table, network_output, row, column)
            batch += 1
