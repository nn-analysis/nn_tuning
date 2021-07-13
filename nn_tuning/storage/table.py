import os
import pickle

from tqdm import tqdm

from .database import Database
from .helpers import __keytolist__, __slicetolist__
from .error import *
import numpy as np


class Table:
    """
    Class containing the table structure.
    The table stores each row in separate files on the machine to offload ram.
    Since file structure are not well equipped to handle a (very) large amount of files,
    do keep that in mind when initialising tables and transpose the data if necessary.

    Tables store rows with other dimensions removed.

    Slicing
    -----------
    Tables can be accessed using slicing. Slicing in Tables works similar to slicing in Numpy arrays.

    Slicing support both get, set, and delete commands.

    Examples
    -----------
    >>> table[1,2]
    3 <-- this is the element in the second row, in the third column of the `Table`.
    >>> table[1,2:4]
    Array([3, 4]) <-- The second and third column of the second row.
    >>> table[1:5]
    Array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]) <-- The rows from the second row to the fifth row in a `Table` with 4 columns.


    Args:
        name: The name of the `Table`
        database: The `Database` the Table resides in
        table_set: The `TableSet` that is the parent of this `Table`
    """

    __properties_file: str
    name: str
    database: Database
    verbose: bool

    def __init__(self, name: str, database: Database, table_set=None):
        self.__properties_file: str = 'properties'
        self.name = name
        self.database = database
        self.table_set = table_set
        self.verbose = False
        self.__nrows = None
        self.__ncols = None
        self.__dtype = None
        self.inputs = None
        self.outputs = None

    @property
    def folder(self):
        if self.table_set is not None:
            return self.table_set.folder + self.name + '/'
        return self.database.folder + self.name + '/'

    @property
    def shape(self) -> tuple:
        """The shape of the table"""
        return self.nrows, self.ncols

    @property
    def nrows(self) -> int:
        """The number of rows in the table"""
        if self.__nrows is None:
            self.__calc_properties__()
        return self.__nrows

    @property
    def ncols(self) -> int:
        """The number of columns in the table"""
        if self.__ncols is None:
            self.__calc_properties__()
        return self.__ncols

    @property
    def dtype(self) -> np.dtype:
        """The datatype of the table"""
        if self.__dtype is None:
            self.__calc_properties__()
        return self.__dtype

    @property
    def initialised(self) -> bool:
        """Indicates whether the necessary files are initialised and whether the shapes are correct"""
        properties_exist = os.path.isfile(self.folder + self.__properties_file)
        if not properties_exist:
            return False
        try:
            with open(self.folder + str(self.__properties_file), 'rb') as f:
                nrows, ncols, inputs, outputs = pickle.load(f)
                nrows = int(nrows)
                ncols = int(ncols)
        except EOFError:
            return False
        except TypeError:
            return False
        except ValueError:
            try:
                self.__nrows, self.__ncols = pickle.load(f)
                self.__nrows = int(self.__nrows)
                self.__ncols = int(self.__ncols)
                self.inputs, self.outputs = None, None
                self.__update_properties__()
                return self.initialised
            except ValueError:
                return False
        try:
            last_row = self.__readfile__(nrows-1, override=True)
        except FileNotFoundError:
            return False
        except EOFError:
            return False
        if type(last_row) is not np.ndarray:
            return False
        if inputs is not None and type(inputs) is not list:
            return False
        if outputs is not None and type(outputs) is not list:
            return False
        return last_row.size == ncols

    def __calc_properties__(self):
        """Calculates the properties of the table including the nrows and ncols"""
        self.__nrows, self.__ncols, self.inputs, self.outputs = self.__readfile__(self.__properties_file)
        self.__nrows = int(self.__nrows)
        self.__ncols = int(self.__ncols)
        self.__dtype = self.__readfile__('0').dtype

    def __update_properties__(self):
        """Updates the properties of the table including the nrows and ncols"""
        self.__writefile__(self.__properties_file, (self.__nrows, self.__ncols, self.inputs, self.outputs), override=True)

    def change_dtype(self, dtype: np.dtype):
        """
        Changes the dtype of the table

        Args:
            dtype: The desired `np.dtype`
        """
        for row in tqdm(range(0, self.nrows), disable=(not self.verbose), leave=False):
            self.__writefile__(row, self.__readfile__(row).astype(dtype))

    def initialise(self, data: np.ndarray, dtype: np.dtype = None, inputs: list = None, outputs: list = None):
        """
        Initialises the table using the np.ndarray provided

        Examples
        ---------
        >>> Table('Name', Database('path')).initialise(np.array([[1,2,3,4],[2,3,4,5]]))
        Creates a table with name 'Name' and two rows of data Array([[1,2,3,4], [2,3,4,5]])

        Args:
            data: The array containing the data for the Table
            dtype (optional): The dtype of the data, changes the data's dtype if they don't match
            inputs (optional): Denotes the inputs for the data in this Table represented as a list of Table names. Table names have to be in the same TableSet or Database.
            outputs (optional): Denotes the outputs for the data in this Table represented as a list of Table names. Table names have to be in the same TableSet or Database.
        """
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder, 0o755)
        # Make data 2 dimensional
        data = data.reshape((data.shape[0], -1))
        # Change dtype if necessary
        if np.dtype is not None:
            data = data.astype(dtype)
        # Go through all the data with a progress bar
        i = 0
        for row in tqdm(data, disable=(not self.verbose), leave=False):
            self.__writefile__(i, row, override=True)
            i += 1
        self.__nrows = data.shape[0]
        self.__ncols = data.shape[1]
        self.inputs = inputs
        self.outputs = outputs
        self.__update_properties__()

    def append_rows(self, data: np.ndarray):
        """
        Add a new row to the existing Table

        Examples
        ---------
        >>> Table('Name', Database('path')).append_rows(np.array([[10,11,12,13], [11, 12, 13, 14]]))
        Adds rows Array([[10,11,12,13], [11, 12, 13, 14]]) to the existing table at path 'path' with name 'Name'.

        Args:
            data: The new rows as an np.ndarray
        """
        reshaped_data = data.reshape(data.shape[0], -1).astype(self.dtype)
        if reshaped_data.shape[1] != self.ncols:
            raise ValueError('Expected data.shape[1] to match self.ncols!')
        i = self.nrows
        for row in reshaped_data:
            self.__writefile__(i, row)
            i += 1
        self.__nrows += reshaped_data.shape[0]
        self.__update_properties__()

    def __repr__(self):
        return f"Table('{self.name}', {self.shape}, '{self.folder}')"

    def __getitem__(self, key):
        rows, cols = __keytolist__(key, self.nrows)
        results = []
        # Go through all rows that need to be fetched and display a progress bar
        for row in tqdm(rows, disable=(not self.verbose), leave=False):
            # Try to fetch the columns and add them to the results
            try:
                if cols is None:
                    results.append(self.__readfile__(row))
                else:
                    results.append(np.array(self.__readfile__(row)[cols]))
            except OSError:
                raise IndexError(f"index {row} is out of bounds for axis 0 with size {self.nrows}")
        # Make the result an np.array again and return them as a list if there is more than one item, else return that item
        results = np.array(results)
        if results.shape[0] is 1:
            return results[0]
        return results

    def __setitem__(self, key, value):
        rows, cols = __keytolist__(key, self.nrows)
        cols_is_slice = False
        cols_len = 1
        if type(cols) is slice and cols.start is None and cols.step is None and cols.stop is None and type(
                value) is np.ndarray:
            cols = None
            expected_shape = (len(rows),)
        elif type(cols) is slice or type(cols) is list:
            cols_is_slice = True
            cols_len = len(__slicetolist__(cols, self.ncols))
            expected_shape = (len(rows), cols_len)
        else:
            expected_shape = (len(rows),)
        i = 0
        for row in rows:
            if not os.path.isfile(self.folder + str(row)):
                raise IndexError(f"index {row} is out of bounds for axis 0 with size {self.nrows}")
            if cols is None:
                if type(value) is not np.ndarray:
                    raise ValueError(f"expected value of type np.ndarray, got {type(value)}")
                if value.ndim == 1:
                    if value.shape[0] != self.ncols:
                        raise ShapeMismatchError(f"expected {self.ncols} columns, found {value.shape[0]}")
                    updated = value
                elif value.ndim == 2:
                    if value.shape[0] != len(rows):
                        raise ShapeMismatchError(f"expected {len(rows)} rows, got {value.shape[0]}")
                    if value.shape[1] != self.ncols:
                        raise ShapeMismatchError(f"expected {self.ncols} columns, found {value.shape[1]}")
                    updated = value[i]
                else:
                    raise ShapeMismatchError(f"expected array of <=2 dimensions, got {value.ndim}")
            else:
                if type(value) is np.ndarray:
                    if value.ndim == 1 and cols_is_slice:
                        if value.shape[0] != cols_len:
                            raise ShapeMismatchError(f"expected array of shape ({cols_len},), got {value.shape}")
                        value_to_set = value
                    elif value.ndim == 1 or value.ndim == 2:
                        if value.shape != expected_shape:
                            raise ShapeMismatchError(f"expected array of shape {expected_shape}, got {value.shape}")
                        value_to_set = value[i]
                    else:
                        raise ShapeMismatchError(f"expected array of <=2 dimensions, got {value.ndim}")
                else:
                    value_to_set = value
                updated = self.__readfile__(row)
                updated[cols] = value_to_set
            self.__deletefile__(row)
            self.__writefile__(row, updated, override=True)
            i += 1

    def __delitem__(self, key):
        rows, cols = __keytolist__(key, self.nrows)
        if cols is not None:
            raise ValueError("Table does not support deleting columns")
        old_nrows = self.nrows
        for row in rows:
            try:
                self.__deletefile__(row)
                self.__nrows -= 1
            except OSError:
                removed = 0
                for key in range(0, old_nrows):
                    if os.path.isfile(self.folder + str(key)):
                        os.rename(self.folder + str(key), self.folder + str(key - removed))
                    else:
                        removed += 1
                self.__update_properties__()
                raise IndexError(f"index {row} is out of bounds for axis 0 with size {self.nrows}")
        removed = 0
        for key in range(0, old_nrows):
            if os.path.isfile(self.folder + str(key)):
                os.rename(self.folder + str(key), self.folder + str(key - removed))
            else:
                removed += 1
        self.__update_properties__()

    def __readfile__(self, filename, override=False):
        if not override and not self.initialised:
            raise TableNotInitialisedError("Initialise the table before using it!")
        with open(self.folder + str(filename), 'rb') as f:
            return pickle.load(f)

    def __writefile__(self, filename, value, override=False):
        if not override and not self.initialised:
            raise TableNotInitialisedError("Initialise the table before using it!")
        with open(self.folder + str(filename), 'wb') as f:
            pickle.dump(value, f)

    def __deletefile__(self, file):
        if not self.initialised:
            raise TableNotInitialisedError("Initialise the table before using it!")
        os.remove(self.folder + str(file))
