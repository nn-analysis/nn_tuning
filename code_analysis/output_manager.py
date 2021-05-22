import numpy as np
from tqdm import tqdm

from .input_manager import InputManager
from code_analysis.networks.network import Network
from .storage import StorageManager


class OutputManager:
    """
    The OutputManager is the class that goes through batches of input.
    The batches are retrieved from the provided InputManager.
    The results are stored using the provided StorageManager.

    Attributes:
        network: The network that will be used.
        storage_manager: The StorageManager that will allow for saving the results.
        input_manager: The InputManager that will provide the input

    Args:
        network: The network that will be used.
        storage_manager: The StorageManager that will allow for saving the results.
        input_manager: The InputManager that will provide the input
    """

    def __init__(self, network: Network, storage_manager: StorageManager, input_manager: InputManager):
        self.network = network
        self.storage_manager = storage_manager
        self.input_manager = input_manager

    def run(self, table: str, batch_size: int, override: bool = True, resume: bool = False, verbose: bool = False):
        """
        Function runs a batch through a network

        Args:
            verbose: outputs the current batch and total percentage done
            table: Table to save data to
            batch_size: Size of the batches to input into the network
            override: bool determines whether the table gets extended or overridden (default=False)
            resume: resume at last batch on failure
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
        batch_end = batch
        while self.input_manager.valid(batch_end, batch_size):
            batch_end += 1
        for _ in tqdm(range(batch, batch_end), disable=(not verbose)):
            self.network.current_batch = batch
            network_input = self.input_manager.get(batch, batch_size)
            network_output, network_output_labels = self.network.run(network_input)
            self.storage_manager.save_result_table_set(network_output, table, network_output_labels, append_rows=True)
            batch += 1
