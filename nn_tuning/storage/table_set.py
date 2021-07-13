import os
import pickle
from typing import Union

import numpy as np
from tqdm import tqdm

from .database import Database
from .error import NoSuchTableError, TableNotInitialisedError
from .helpers import __keytolist__, __slicetolist__, __verify_data_types_are_correct__
from .table import Table


class TableSet:
    """
    A set of `Table`s and or other `TableSet`s.
    The get, set, and delete functions work on the aggregated table data.
    To get a specific subtable you can use `get_subtable(key)`

    Slicing
    -----------
    TableSets can be accessed using slicing. Slicing in TableSets works similar to slicing in Numpy arrays.

    The `TableSet` slicing combines all subtables into one structure. When using slicing the underlying subtables are combined.

    Slicing support both get, set, and delete commands.

    Examples
    -----------
    >>> tableset[1,2]
    3 <-- this is the element in the second row, in the third column of the `TableSet`.
    >>> tableset[1,2:4]
    Array([3, 4]) <-- The second and third column of the second row.
    >>> tableset[1:5]
    Array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]) <-- The rows from the second row to the fifth row in a `TableSet` with 4 columns.

    Attributes:
        name: Name of the `TableSet`.
        database: The `database` the TableSet is in.
        table_set (optional): The TableSet this TableSet is a part of.
        verbose: If true progress bars are shown during some operations

    Args:
        name: Name of the TableSet.
        database: The database the TableSet is in.
        table_set (optional): The TableSet this TableSet is a part of.
        verbose (optional, default=False): If true progress bars are shown during some operations
    """

    name: str
    database: Database
    verbose: bool

    def __init__(self, name: str, database: Database, table_set=None, verbose=False):
        self.name = name
        self.database = database
        self.table_set = table_set
        self.__properties_file = 'properties'
        self.verbose = verbose
        self.__subtables = None
        self.__dtype = None
        self.__ncols = None
        self.__ncols_tuple = None
        self.__recurrent_subtables = None
        self.inputs = None
        self.outputs = None

    def __repr__(self):
        return f"TableSet('{self.name}', ({self.nrows}, {self.ncols}), '{self.folder}')"

    def __getitem__(self, key):
        rows, cols = __keytolist__(key, self.nrows)
        results = []
        for row in tqdm(rows, disable=(not self.verbose), leave=False):
            try:
                subtable_total = None
                for subtables in self.subtables:
                    if subtable_total is None:
                        subtable_total = self.get_subtable(subtables)[row]
                    else:
                        subtable_total = np.concatenate((subtable_total, self.get_subtable(subtables)[row]))
                if cols is None:
                    results.append(subtable_total)
                else:
                    results.append(subtable_total[cols])
            except OSError:
                raise IndexError(f"index {row} is out of bounds for axis 0 with size {self.nrows}")
        results = np.array(results)
        if results.shape[0] is 1:
            return results[0]
        return results

    def __setitem__(self, key, value):
        rows, cols = __keytolist__(key, self.nrows)

        # Make sure the shape and types of the keys and the values is something we expect
        single_value = False
        len_value = 1
        if type(value) is list:
            value = np.array(value)
            if value.ndim > 1:
                value = value.reshape(len(value), -1)
        if type(value) is np.ndarray:
            if value.ndim > 1:
                len_value = value.shape[1]
            else:
                len_value = value.shape[0]
        if type(cols) is slice and cols.start is None and cols.step is None and cols.stop is None and type(
                value) is np.ndarray:
            cols = None
            if type(value) is np.ndarray:
                if len_value != self.ncols:
                    raise ValueError(f'Expected len(value)={self.ncols}, found {len(value)}')
            else:
                single_value = True
        elif type(cols) is slice:
            cols_len = len(__slicetolist__(cols, self.ncols))
            if len_value != cols_len:
                raise ValueError(f'Expected len(value)={cols_len}, found {len(value)}')
        elif type(cols) is list:
            cols_len = len(cols)
            if len_value != cols_len:
                raise ValueError(f'Expected len(value)={cols_len}, found {len(value)}')
        else:
            single_value = True
            if len_value > 1:
                raise ValueError(f'Expected len(value)=1, found len(value)={len_value}')

        # Go through the subtables to delegate the values to the correct subtables
        min_col = 0
        for subtable in self.subtables:
            subtable_instance = self.get_subtable(subtable)
            max_col = min_col + subtable_instance.ncols-1
            # If only one value is changed check if that column is in the subtable and update it accordingly
            if single_value:
                if min_col <= cols <= max_col:
                    subtable_instance[rows[0], cols] = value
            else:
                # Select the columns that are in this subtable
                cols_array = np.array(__slicetolist__(cols, self.ncols))
                cols_array = cols_array[np.where(cols_array >= min_col)]
                cols_array = cols_array[np.where(cols_array <= max_col)]
                value_array = value
                # Select just the values for the columns in this subtable
                if value_array.ndim > 1:
                    value_array = value_array[:, np.where(cols_array >= min_col)[0]]
                    value_array = value_array[:, np.where(cols_array <= max_col)[0]]
                else:
                    value_array = value_array[np.where(cols_array >= min_col)[0]]
                    value_array = value_array[np.where(cols_array <= max_col)[0]]
                # Make sure the columns start at the right place
                cols_array = cols_array - min_col
                # Make the columns a list and update the subtable with the values and columns that were just calculated
                cols_list = cols_array.tolist()
                if len(cols_list) > 0:
                    subtable_instance[rows, cols_list] = value_array
            min_col += subtable_instance.ncols

    def __delitem__(self, key):
        for subtable in self.subtables:
            del self.get_subtable(subtable)[key]

    def __verify_data_and_names_have_matching_shapes(self, data: tuple, names: dict) -> bool:
        """
        Checks whether the data and names have the same size

        Args:
            data: The data of the subtables
            names: The names of the subtables and subtablesets

        Returns:
            True if they share the same size, False otherwise
        """
        if len(data) != len(names):
            return False
        i = 0
        for subdata in data:
            name = list(names.items())[i]
            if type(subdata) is tuple:
                if type(name[1]) is dict:
                    if not self.__verify_data_and_names_have_matching_shapes(subdata, name[1]):
                        return False
            i += 1
        return True

    def __verify_coherent_data_rows(self, data: tuple) -> bool:
        """
        Checks whether the data has a coherent amount of rows everywhere

        Args:
            data: The data of the subtables

        Returns:
            True if it does, False otherwise
        """
        nrows = None
        for subdata in data:
            if type(subdata) is tuple:
                if not self.__verify_coherent_data_rows(subdata):
                    return False
            elif nrows is None:
                nrows = subdata.shape[0]
            elif nrows != subdata.shape[0]:
                return False
        return True

    def __verify_name_types_are_correct(self, names: dict) -> bool:
        """
        Checks whether the names is a dict of dicts or strings everywhere

        Args:
            names: The names of the subtables and subtablesets

        Returns:
            True if it does, False otherwise
        """
        for name in names.items():
            if type(name[1]) is dict:
                if not self.__verify_name_types_are_correct(name[1]):
                    return False
            elif type(name[0]) is not str:
                return False
        return True

    def initialise(self, data: tuple, names: dict, dtype: np.dtype = None, relations: dict = None, inputs: list = None, outputs: list = None):
        """
        Initialise the TableSet with the structure set in the data parameter.
        Names and data need to be of the same shape.

        The program will use the key of the names dict as the name of the Table or TableSet.

        It is possible to create sub TableSets by making nested dicts in the names variable and nested tuples in the
        data variable.

        Examples
        -----------
        >>> TableSet('Name', Database('path')).initialise((np.array([[1,2,3]]), (np.array([[2,3,4]]))), {'first_subtable':'', 'second_subtable':{'first_subtable':''}})
        Initiliases a TableSet in the database with the given data and names
        >>> TableSet('Name').initialise((np.array([[1,2,3]]), (np.array([[2,3,4]]), np.array([[3,4,5]]))), {'first_subtable':'', 'second_subtable':{'first_subtable':'', 'second_subtable':''}}, relations={'first_subtable':([], ['second_subtable']), 'second_subtable':(['first_subtable'],[],{'first_subtable':([],['second_subtable']), 'second_subtable':(['first_subtable'], [])})})
        Initialises a TableSet in the database with the given data and names that has defined inputs and outputs

        Args:
            data: The data of the subtables
            names: The names of the subtables and subtablesets
            dtype (optional): Data type of the array
            relations (optional): Relations between components of the data in terms of inputs and outputs. Example: `{ 'name': (inputs=list of names, outputs=list of names, child=child dict),  'name2': (inputs=list of names, outputs=list of names, child=child dict), etc... }`
            inputs (optional): Denotes the inputs for the data in this TableSet represented as a list of Table names. TableSet names have to be in the same TableSet or Database.
            outputs (optional): Denotes the outputs for the data in this TableSet represented as a list of Table names. TableSet names have to be in the same TableSet or Database.
        """
        if not __verify_data_types_are_correct__(data):
            raise ValueError('data is not a tuple of nested tuples of np.ndarrays or a tuple of np.ndarrays!')
        if not self.__verify_name_types_are_correct(names):
            raise ValueError('names is not a dict of nested dicts of strings or a dict of strings!')
        if not self.__verify_data_and_names_have_matching_shapes(data, names):
            raise ValueError('data and names do not have the same shape!')
        if not self.__verify_coherent_data_rows(data):
            raise ValueError('expected the rows of all np.ndarrays in the data variable to be of the same length!')
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder, 0o755)
        i = 0
        self.__subtables = []
        self.inputs = inputs
        self.outputs = outputs
        for item in data:
            name = list(names.items())[i][0]
            self.__subtables.append(name)
            inputs = relations[name][0] if relations is not None and name in relations and len(relations[name]) > 0 else None
            outputs = relations[name][1] if relations is not None and name in relations and len(relations[name]) > 1 else None
            child = relations[name][2] if relations is not None and name in relations and len(relations[name]) > 2 else None
            if type(item) is tuple:
                TableSet(name, self.database, self, self.verbose).initialise(item, list(names.items())[i][1], dtype,
                                                                             child, inputs, outputs)
            elif type(item) is np.ndarray:
                Table(name, self.database, self).initialise(item, dtype, inputs, outputs)
            else:
                raise ValueError('Expected type')
            i += 1
        self.__update_properties__()

    def __verify_ncols(self, data: tuple):
        i = 0
        for subdata in data:
            subtable = self.get_subtable(i)
            if type(subdata) is tuple:
                if not subtable.__verify_ncols(subdata):
                    return False
            else:
                if subtable.ncols != subdata.shape[1]:
                    return False
            i += 1
        return True

    def append_rows(self, data: tuple, skip_verification: bool = False):
        """
        Add a new rows to the existing TableSet

        Examples
        -----------
        >>> TableSet('Name', Database('path')).append_rows((np.array([[1,2,3,3]]),))
        Adds a single row to a TableSet with one subtable with 4 columns

        Args:
            data: The new rows as a (nested) tuple of np.ndarrays
            skip_verification (optional, default=False) : If True the verification steps are skipped. This allows for faster processing and is used when this function calls itself.
        """
        # Verify the data
        if not skip_verification:
            if not __verify_data_types_are_correct__(data):
                raise ValueError('data is not a tuple of nested tuples of np.ndarrays or a tuple of np.ndarrays!')
            if not self.__verify_coherent_data_rows(data):
                raise ValueError('expected the rows of all np.ndarrays in the data variable to be of the same length!')
            if not self.__verify_ncols(data):
                raise ValueError('make sure data.ncols is the same as the existing ncols')
        i = 0
        for subtable_key in self.subtables:
            subtable = self.get_subtable(subtable_key)
            if type(subtable) is TableSet:
                subtable.append_rows(data[i], skip_verification=True)
            else:
                subtable.append_rows(data[i])
            i += 1

    @property
    def shape(self) -> (int, int):
        """Shape of the TableSet"""
        return self.nrows, self.ncols

    @property
    def nrows(self) -> int:
        """Amount of rows in this subtable"""
        return self.get_subtable(0).nrows

    @property
    def ncols(self) -> int:
        """Total amount of columns in this subtable"""
        if self.__ncols is not None:
            return self.__ncols
        total = 0
        for subtable in self.subtables:
            total += self.get_subtable(subtable).ncols
        self.__ncols = total
        return total

    @property
    def folder(self) -> str:
        if self.table_set is not None:
            return self.table_set.folder + self.name + '/'
        return self.database.folder + self.name + '/'

    @property
    def subtables(self) -> list:
        """The names of the subtables in this TableSet"""
        if self.__subtables is None:
            self.__calc_properties__()
        return self.__subtables

    @property
    def recurrent_subtables(self) -> dict:
        """Dictionary of names of subtables and subsubtables etc."""
        if self.__recurrent_subtables is not None:
            return self.__recurrent_subtables
        result = {}
        for subtable in self.subtables:
            subtable_instance = self.get_subtable(subtable)
            if type(subtable_instance) is TableSet:
                result[subtable] = subtable_instance.recurrent_subtables
            else:
                result[subtable] = subtable
        self.__recurrent_subtables = result
        return result

    @property
    def ncols_tuple(self) -> tuple:
        """Amount of columns split up by subtable"""
        if self.__ncols_tuple is not None:
            return self.__ncols_tuple
        ncols_tuple = []
        for subtable in self.subtables:
            subtable_instance = self.get_subtable(subtable)
            if type(subtable_instance) is TableSet:
                ncols_tuple.append(self.get_subtable(subtable).ncols_tuple)
            else:
                ncols_tuple.append(subtable_instance.ncols)
        self.__ncols_tuple = tuple(ncols_tuple)
        return self.__ncols_tuple

    @property
    def dtype(self) -> np.dtype:
        """The datatype of the TableSet"""
        if self.__dtype is None:
            self.__calc_properties__()
        return self.__dtype

    def change_dtype(self, dtype: np.dtype):
        """
        Changes the dtype of the TableSet

        Args:
            dtype: The desired dtype
        """
        for subtable in tqdm(self.subtables, disable=(not self.verbose), leave=False):
            self.get_subtable(subtable).change_dtype(dtype)

    def __calc_properties__(self):
        """Calculates the properties of the table including the nrows and ncols"""
        if not self.initialised:
            raise TableNotInitialisedError
        self.__subtables, self.inputs, self.outputs = self.__readfile__(self.__properties_file)
        self.__dtype = self.get_subtable(0).dtype

    def __update_properties__(self):
        self.__writefile__(self.__properties_file, (self.__subtables, self.inputs, self.outputs), override=True)

    @property
    def initialised(self) -> bool:
        """Indicates whether the TableSet was (correctly) initialised"""
        properties_exist = os.path.isfile(self.folder + self.__properties_file)
        if not properties_exist:
            return False
        try:
            with open(self.folder + str(self.__properties_file), 'rb') as f:
                subtables, inputs, outputs = pickle.load(f)
        except EOFError:
            return False
        except TypeError:
            return False
        except ValueError:
            try:
                self.__subtables = pickle.load(f)
                self.inputs, self.outputs = None, None
                self.__update_properties__()
                return self.initialised
            except ValueError:
                return False
        if type(subtables) is not list:
            return False
        if inputs is not None and type(inputs) is not list:
            return False
        if outputs is not None and type(outputs) is not list:
            return False
        return True

    def print_structure(self, tabs=0):
        """
        Prints a map of the table structure

        Args:
            tabs: (Integer) Amount of tabs to print before the text
        """
        raise NotImplementedError()

    def get_subtable(self, key: Union[str, int]):
        """
        Returns the `Table` or `TableSet` with the key

        Args:
            key: The key of the subtable as a string or integer
        """
        if not self.initialised:
            raise TableNotInitialisedError("Initialise the TableSet before calling this function!")
        if type(key) is not int and key not in self.subtables:
            raise NoSuchTableError()
        if type(key) is int:
            key = self.subtables[key]
        if Table(key, self.database, self).initialised:  # Subtable is a Table
            return Table(key, self.database, self)
        elif TableSet(key, self.database, self).initialised:  # Subtable is a TableSet
            return TableSet(key, self.database, self)
        raise NoSuchTableError('No subtable with this key was found to be initialised!')

    def __readfile__(self, filename, override=False):
        if not override and not self.initialised:
            raise TableNotInitialisedError("Initialise the TableSet before using it!")
        with open(self.folder + str(filename), 'rb') as f:
            return pickle.load(f)

    def __writefile__(self, filename, value, override=False):
        if not override and not self.initialised:
            raise TableNotInitialisedError("Initialise the TableSet before using it!")
        with open(self.folder + str(filename), 'wb') as f:
            pickle.dump(value, f)
