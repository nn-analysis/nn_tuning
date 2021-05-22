import os
import shutil

import numpy as np

from .table_set import TableSet
from .table import Table
from .error import NoSuchTableError
from .database import Database
from .helpers import __verify_data_types_are_correct__


class StorageManager:
    """
    Class handling often used storage related queries such as saving results from an experiment or from fitting to a table_set

    Args:
        database (str) : Database the Tables and TableSets should reside in
    """

    database: Database

    def __init__(self, database: Database):
        self.database = database

    def save_result_table_set(self, results: tuple, name: str, table_labels: dict, nrows: int = None,
                              ncols: tuple = None, row_start: int = 0, col_start: int = 0,
                              append_rows: bool = False) -> TableSet:
        """
        Function that saves a TableSet with the results.
        Results do not have to be complete. If the result is incomplete provide the nrows or ncols parameters.
        The lacking rows or columns will then be filled with zeros. When providing the additional results use the
        row_start and col_start parameters to indicate where the existing TableSet is to be overridden.

        If a result tuple is incomplete and has completely missing parts fill in the missing parts with None values.

        Args:
            results (tuple): A (nested) tuple of np.arrays containing results
            name (str): The name of the TableSet to be
            table_labels (dict): A (nested) tuple of the names of the Tables in the TableSet
            nrows (int, optional): Amount of rows that will (eventually) be in the table set. If not provided nrows=results.shape[0]
            ncols (tuple, optional): Amount of columns that will (eventually) be in the table set per subpart of the data. If not provided ncols[i]=results[i].shape[1]
            row_start (int, optional): Int of the row where the results start. This parameter is used if nrows does not match the amount of rows in the results.
            col_start (int, optional): Int of the column where the results start. This parameter is used if ncols does not match the amount of columns in the results.
            append_rows (bool, optional) : If true, when the TableSet is already initialised the results will be appended as rows to the existing TableSet

        Returns:
            object (`TableSet`) : TableSet containing the results
        """
        table_set = TableSet(name, self.database)
        if not __verify_data_types_are_correct__(results):
            raise ValueError('results is not a tuple of nested tuples of np.arrays or a tuple of np.arrays!')
        if table_set.initialised:  # Update existing TableSet
            if append_rows:
                table_set.append_rows(results)
            else:
                # Combine the small arrays into one large array so we can use the TableSet update function
                combined_results = self.__combine_into_big_array(results)
                table_set[row_start:combined_results.shape[0]+row_start,
                          col_start:combined_results.shape[1]+col_start] = combined_results
        else:
            if self.__requires_nrows_calculation(results, nrows):
                nrows = self.__fetch_nrows_from_partial_result_set(results)
            if ncols is None and self.__requires_ncols_to_be_set_somewhere(results):
                raise ValueError('When one of the results is None, ncols has to be set for that result!')
            padded_results = self.__pad_with_zeros(results, ncols, nrows)
            table_set.initialise(padded_results, table_labels)
        return table_set

    def open_table(self, name: str):
        """
        Opens a Table or TableSet in the Database with the given name.
        The function can automatically distinguish between Tables and TableSet.

        Args:
            name: Name of the Table or TableSet

        Returns:
            Table or TableSet
        """
        if Table(name, self.database).initialised:
            return Table(name, self.database)
        elif TableSet(name, self.database).initialised:
            return TableSet(name, self.database)
        else:
            raise NoSuchTableError

    def remove_table(self, name: str):
        """
        Deletes a table from the Database

        Args:
            name: The name of the table
        """
        if os.path.isdir(self.database.folder + name):
            shutil.rmtree(self.database.folder + name)

    def __requires_nrows_calculation(self, results: tuple, nrows: int = None) -> bool:
        if nrows is not None:
            return False
        for result in results:
            if type(result) is tuple:  # If the result is a tuple unpack it further
                if not self.__requires_nrows_calculation(results):
                    return False
            elif result is None:
                return True
        return False

    def __requires_ncols_to_be_set_somewhere(self, results: tuple):
        for result in results:
            if type(result) is tuple:  # If the result is a tuple unpack it further
                if not self.__requires_nrows_calculation(results):
                    return False
            elif result is None:
                return True
        return False

    def __fetch_nrows_from_partial_result_set(self, results: tuple) -> int:
        for result in results:
            if type(result) is tuple:
                nrows_subresult = self.__fetch_nrows_from_partial_result_set(result)
                if nrows_subresult is not None:
                    return nrows_subresult
            elif result is not None:
                return result.shape[0]
        raise ValueError('When all results are None values nrows has to be set!')

    def __pad_with_zeros(self, results: tuple, ncols: tuple, nrows: int) -> tuple:
        i = 0
        final_result = []
        for result in results:
            if type(result) is tuple:  # If the result is a tuple unpack it further
                final_result.append(self.__pad_with_zeros(result, ncols[i], nrows))
            elif result is None:  # If the result is None fill the result using the ncols and nrows
                if ncols[i] is None:
                    raise ValueError('When a particular result is None ncols for that result has to be set!')
                final_result.append(np.zeros((nrows, ncols[i])))
            else:  # If the result is a result fit it in a padded result array of shape (ncols, nrows)
                flattened_array = result.reshape(result.shape[0], -1)
                this_nrows = flattened_array.shape[0]
                if nrows is not None:
                    this_nrows = nrows
                padded_array = np.zeros((this_nrows, ncols[i]))
                padded_array[:, :flattened_array.shape[1]] = flattened_array
                final_result.append(padded_array)
            i += 1
        return tuple(final_result)

    def __combine_into_big_array(self, results: tuple) -> np.array:
        final_result = []
        for result in results:
            if type(result) is tuple:
                final_result.append(self.__combine_into_big_array(result))
            elif result is None:
                continue
            else:
                final_result.append(result)
        return np.concatenate(final_result)
