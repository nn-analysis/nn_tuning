import pickle
import numpy as np
import os.path
import glob
import shutil


class NoSuchTableError(Exception):
    pass


class ShapeMismatchError(Exception):
    pass


class TableAlreadyExists(Exception):
    pass


class Table:

    name: str
    database_folder: str
    __properties_file: str
    __table_location: str
    __nrows: int = None
    __ncols: int = None
    __index: list = None
    __cols: list = None
    __dtype: np.dtype = None

    def __init__(self, name: str, folder: str, verbose: bool = False):
        if folder[-1] == "/":
            folder = folder[:-1]
        folder += "/"
        self.__properties_file = 'properties'
        self.name = name
        self.database_folder = folder
        self.__table_location = folder + name + "/"
        self.verbose = verbose

    def __repr__(self):
        return f"Table('{self.name}', {self.shape}, '{self.database_folder}')"

    @property
    def shape(self):
        return self.nrows, self.ncols

    @property
    def nrows(self):
        if self.__nrows is None:
            self.__calcproperties__()
        return int(self.__nrows)

    @property
    def ncols(self):
        if self.__ncols is None:
            self.__calcproperties__()
        return int(self.__ncols)

    @property
    def row_index(self):
        if self.__index is None:
            self.__calcproperties__()
        return self.__index

    @property
    def dtype(self):
        if self.__dtype is None:
            self.__calcproperties__()
        return self.__dtype

    @property
    def column_index(self):
        if self.__cols is None:
            self.__calcproperties__()
        return self.__cols

    def __calcproperties__(self):
        if not self.initialised:
            raise NoSuchTableError
        self.__nrows, self.__ncols, self.__index, self.__cols = self.__readfile__(self.__properties_file)
        self.__dtype = self.__readfile__('0').dtype

    def __updateproperties__(self):
        self.__writefile__(self.__properties_file, (self.__nrows, self.__ncols, self.__index, self.__cols))

    def row_from_key(self, key) -> int:
        if self.__index is None:
            self.__calcproperties__()
        return self.__index.index(key)

    def col_from_key(self, key) -> int:
        if self.__cols is None:
            self.__calcproperties__()
        return self.__cols.index(key)

    def __getitems__(self, keys: tuple):
        rows, cols = keys
        results = []
        i = 0
        for row in rows:
            if self.verbose:
                print(f"{i}/{len(rows)}, {int(i/len(rows)*100)}%\r", end="")
            try:
                if cols is None:
                    results.append(self.__readfile__(row))
                else:
                    results.append(np.array(self.__readfile__(row)[cols]))
            except OSError:
                raise IndexError(f"index {row} is out of bounds for axis 0 with size {self.nrows}")
            i += 1
        if self.verbose:
            print()
        results = np.array(results)
        if results.shape[0] is 1:
            return results[0]
        return results

    def __slicetolist__(self, key, length: int = None):
        if length is None:
            length = self.nrows
        start = key.start
        if start is None:
            start = 0
        stop = key.stop
        if stop is None:
            stop = 0
        if start == 0 and stop == 0:
            stop = length
        step = key.step
        if step is None:
            step = 1
        return list(range(start, stop, step))

    def __keytolist__(self, key) -> tuple:
        if not self.initialised:
            raise NoSuchTableError
        if type(key) is slice:
            return self.__slicetolist__(key), None
        elif type(key) is tuple:
            if len(key) > 2:
                raise ShapeMismatchError(f"Expected <=2, got {len(key)}")
            rows, cols = key
            if type(rows) is slice:
                rows = self.__slicetolist__(rows)
            elif type(rows) is np.ndarray or type(rows) is list:
                rows = rows
            else:
                rows = [rows]
            return rows, cols
        elif type(key) is np.ndarray or type(key) is list:
            return key, None
        else:
            return [key], None

    def __getitem__(self, key):
        return self.__getitems__(self.__keytolist__(key))

    def __delitems__(self, keys: tuple):
        rows, cols = keys
        if cols is not None:
            raise ValueError("Table does not support deleting columns")
        old_nrows = self.nrows
        for row in rows:
            try:
                self.__deletefile__(row)
                del self.__index[row]
                self.__nrows -= 1
            except OSError:
                removed = 0
                for key in range(0, old_nrows):
                    if os.path.isfile(self.__table_location + str(key)):
                        os.rename(self.__table_location + str(key), self.__table_location + str(key-removed))
                    else:
                        removed += 1
                self.__updateproperties__()
                raise IndexError(f"index {row} is out of bounds for axis 0 with size {self.nrows}")
        removed = 0
        for key in range(0, old_nrows):
            if os.path.isfile(self.__table_location + str(key)):
                os.rename(self.__table_location + str(key), self.__table_location + str(key-removed))
            else:
                removed += 1
        self.__updateproperties__()

    def __delitem__(self, key):
        self.__delitems__(self.__keytolist__(key))

    def __setitems__(self, keys: tuple, value):
        rows, cols = keys
        cols_is_slice = False
        cols_len = 1
        if type(cols) is slice and cols.start is None and cols.step is None and cols.stop is None and type(value) is np.array:
            cols = None
            expected_shape = (len(rows),)
        elif type(cols) is slice:
            cols_is_slice = True
            cols_len = len(self.__slicetolist__(cols, self.ncols))
            expected_shape = (len(rows), cols_len)
        else:
            expected_shape = (len(rows),)
        i = 0
        for row in rows:
            if not os.path.isfile(self.__table_location + str(row)):
                raise IndexError(f"index {row} is out of bounds for axis 0 with size {self.nrows}")
            if cols is None:
                if type(value) is not np.ndarray:
                    raise ValueError(f"expected value of type np.array, got {type(value)}")
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
            self.__writefile__(row, updated)
            i += 1

    def __setitem__(self, key, value: np.array):
        self.__setitems__(self.__keytolist__(key), value)

    def append(self, item: np.array, row_index):
        self.extend(item[np.newaxis, ...], list(row_index))

    def extend(self, items: np.array, row_index: list):
        if not self.initialised:
            raise NoSuchTableError
        if items.shape[1] != self.ncols:
            raise ShapeMismatchError(f"expected {self.ncols} columns, found {items.shape[1]}")
        if items.dtype is not self.dtype:
            items = items.astype(self.dtype)
        j = self.nrows
        for i in range(0, items.shape[0]):
            if self.verbose:
                print(f"{i}/{items.shape[0]}, {int(i/items.shape[0]*100)}%\r", end="")
            self.__writefile__(str(j), items[i])
            self.__nrows += 1
            j += 1
        self.__index.extend(row_index)
        self.__updateproperties__()

    def change_dtype(self, dtype: np.dtype, as_new_table: str = None):
        tmp_tbl = Table(self.name, self.database_folder, False)
        if as_new_table is None:
            for row in range(self.nrows):
                if self.verbose:
                    print(f"{row}/{self.nrows}, {int(row/self.nrows*100)}%\r", end="")
                tmp_tbl[row] = tmp_tbl[row].astype(dtype)
        else:
            new_table = Table(as_new_table, self.database_folder, False)
            new_table.create(tmp_tbl[0][np.newaxis, ...], self.row_index[0], self.column_index, dtype)
            if self.verbose:
                print(f"{0}/{self.nrows}, {int(0/self.nrows*100)}%\r", end="")
            for row in range(1, self.nrows):
                if self.verbose:
                    print(f"{row}/{self.nrows}, {int(row/self.nrows*100)}%\r", end="")
                new_table.extend(tmp_tbl[row], self.row_index[row])
        print()

    def create(self, array: np.array, row_index: list = None, column_index: list = None,
               shape: (int, int) = None, indices: (slice, slice) = None, dtype: np.dtype = None):
        if not os.path.isdir(self.__table_location):
            os.mkdir(self.__table_location, 0o755)
        array = array.reshape((array.shape[0], -1))
        self.__nrows = shape[0] if shape is not None else array.shape[0]
        self.__ncols = shape[1] if shape is not None else array.shape[1]
        if row_index is None:
            row_index = list(range(self.__nrows))
        if column_index is None:
            column_index = list(range(self.__ncols))
        self.__index = row_index
        self.__cols = column_index
        if len(self.row_index) != self.nrows:
            str_one = "shape[0]" if shape is not None else "item.shape[0]"
            raise ShapeMismatchError(f'expected len(row_index)=={str_one}, got {len(row_index)}!={self.nrows}')
        if len(self.column_index) != self.ncols:
            str_one = "shape[1]" if shape is not None else "item.shape[1]"
            raise ShapeMismatchError(f'expected len(self.column_index)=={str_one}, got {len(self.column_index)}!={self.ncols}')
        indices_y = self.__slicetolist__(indices[0]) if (indices is not None) and (indices[0] is not None) else list(range(0, self.nrows))
        indices_x = self.__slicetolist__(indices[1], length=self.ncols) if (indices is not None) and (indices[1] is not None) else list(range(0, self.ncols))
        if int(array.shape[0]) != int(len(indices_y)):
            raise ShapeMismatchError(f'expected array.shape[0]==len(indices[0]), got {array.shape[0]}!={len(indices_x)}')
        if int(array.shape[1]) != int(len(indices_x)):
            raise ShapeMismatchError(f'expected array.shape[1]==len(indices[1]), got {array.shape[1]}!={len(indices_y)}')
        self.__updateproperties__()
        custom_columns = len(indices_x) == array.shape[1]
        i = 0
        for row in range(0, self.nrows):
            if self.verbose:
                print(f'{row}/{self.nrows}, {int(row/self.nrows*100)}%\r', end="")
            if custom_columns and row in indices_y:
                row_array = np.zeros(self.ncols)
                row_array[indices_x] = array[i]
                i += 1
            elif row in indices_y:
                row_array = array[i]
                i += 1
            else:
                row_array = np.zeros(self.ncols)
            if dtype is not None:
                row_array = row_array.astype(dtype)
            self.__writefile__(row, row_array)
        print()

    @property
    def initialised(self) -> bool:
        return os.path.isfile(self.__table_location + self.__properties_file)

    def T(self):
        return self.transpose()

    def transpose(self, new_table: str = None, batch_size: int = 1, row_batch_size: int = None, override: bool = True):
        verbose = self.verbose
        if not self.initialised:
            raise NoSuchTableError
        if row_batch_size is None:
            row_batch_size = self.nrows
        if new_table is None:
            new_table = self.name + "_T"
        new_table = Table(new_table, self.database_folder, self.verbose)
        if new_table.initialised and not override:
            return new_table
        initialised = False
        for col in range(0, int(self.ncols), batch_size):
            if verbose:
                print(f"{col}/{self.ncols}, {int(col/self.ncols*100)}%")
            col_result = []
            end = col+batch_size if col+batch_size < self.ncols else self.ncols
            if row_batch_size == self.nrows:
                col_result = self[:, col:end].T
            elif row_batch_size == self.nrows:
                row_batches = range(0, self.nrows, row_batch_size)
                for row in row_batches:
                    row_batch_end = row + row_batch_size if row + row_batch_size < self.nrows else self.nrows
                    if verbose:
                        print(f'{int(row/self.nrows*100)}%, {row}/{self.nrows}\r', end="")
                    if row_batch_end - row == 1:
                        col_result.append(np.array(self[row:row_batch_end, col:end][np.newaxis, :]))
                    else:
                        col_result.append(np.array(self[row:row_batch_end, col:end]).T)
                    if len(col_result) > 1 and col_result[-1].shape[0] != col_result[-2].shape[0]:
                        col_result[-1] = col_result[-1].T
                col_result = np.concatenate(col_result, axis=1)
                if verbose:
                    print()
            if col_result.ndim == 1:
                col_result = col_result[np.newaxis, ...]
            if not initialised:
                new_table.create(col_result, row_index=self.__cols, column_index=self.__index)
                initialised = True
            else:
                new_table.extend(col_result, [])
        return new_table

    def __readfile__(self, filename):
        with open(self.__table_location + str(filename), 'rb') as f:
            return pickle.load(f)

    def __writefile__(self, filename, value):
        with open(self.__table_location + str(filename), 'wb') as f:
            pickle.dump(value, f)

    def __deletefile__(self, file):
        os.remove(self.__table_location + str(file))

    @staticmethod
    def shape_to_indices(shape: tuple) -> list:
        arr = np.zeros(shape).reshape(-1)
        arr[:] = list(range(0, arr.shape[0]))
        arr = arr.reshape(shape)
        indices = []
        for i in range(0, arr.size):
            indices.append(tuple([x[0] for x in np.where(arr == i)]))
        return indices

    def calculate_best_fits(self, p, new_tbl: str = None, override: bool = False):
        if new_tbl is None:
            new_tbl = self.name + "_best_results"
        new_tbl = Table(new_tbl, self.database_folder, self.verbose)
        if new_tbl.initialised and not override:
            return new_tbl
        best_fits = np.zeros(self.ncols, dtype=self.dtype)
        p_rows = np.zeros(self.ncols, dtype=np.int)
        row_index = ['best_fit']
        for row in range(0, self.nrows):
            row_values = self.__readfile__(row)
            better_fits = np.where(row_values > best_fits)
            best_fits[better_fits] = row_values[better_fits]
            p_rows[better_fits] = row
        if p is not None:
            best_p_values = p[p_rows]
            best_fits = np.concatenate((best_fits.reshape(1, -1), best_p_values.T))
            row_index.extend(['x', 'y', 's'])
        else:
            best_fits = np.concatenate((best_fits.reshape(1, -1), p_rows))
            row_index.append('location')
        if not self.initialised:
            raise NoSuchTableError
        new_tbl.create(best_fits, column_index=self.column_index, row_index=row_index)
        return new_tbl


class StorageManager:

    _folder: str

    def __init__(self, folder: str):
        self.initialise_database(folder)

    def initialise_database(self, folder: str):
        if folder[-1] == "/":
            folder = folder[:-1]
        self._folder = folder + "/"
        if not os.path.isdir(self._folder):
            os.mkdir(self._folder, 0o755)

    def open_table(self, table: str, verbose=False) -> Table:
        return Table(table, self._folder, verbose=verbose)

    def remove_table(self, table: str):
        if os.path.isdir(self._folder + table):
            shutil.rmtree(self._folder + table)

    def list_tables(self):
        folder_content = glob.glob(self._folder)
        tables = []
        for f in folder_content:
            table = f.split('/')[-1]
            if self.__is_valid_table(table):
                tables.append(table)
        return tables

    def __is_valid_table(self, table: str) -> bool:
        if not os.path.isdir(table) or not os.path.isfile(table + '/properties'):
            return False
        tmp_tbl = Table(table, self._folder)
        properties_content = tmp_tbl.__readfile__('properties')
        if not isinstance(properties_content, tuple) or len(properties_content) < 4:
            return False
        a, b, c, d = properties_content
        if not isinstance(a, int) or not isinstance(b, int) or not isinstance(c, list) or not isinstance(d, list):
            return False
        if len(c) != a or len(d) != b:
            return False
        folder_content = glob.glob(f'{self._folder}/{table}')
        folder_content.remove(f'{self._folder}/{table}/properties')
        if len(folder_content) != a:
            return False
        return True

    def rename_table(self, old_name: str, new_name: str):
        os.rename(self._folder + old_name, self._folder + new_name)

    def create_table(self, table: str, array: np.array, row_index: list = None, column_index: list = None, shape: tuple = None,
                     indices: tuple = None, verbose: bool = False, dtype: np.dtype = None):
        tbl = Table(table, self._folder, verbose)
        if tbl.initialised:
            raise TableAlreadyExists(f'table {table} already exists, please remove the table or choose a different name')
        tbl.create(array, row_index, column_index, shape, indices, dtype)
        return tbl

    def save_results(self, table: str, results: np.array, row_index: list, column_index: list, shape: (int, int) = None,
                     indices: (slice, slice) = None, verbose: bool = False, dtype: np.dtype = None):
        if row_index is None:
            row_index = range(0, results.shape[0])
        if column_index is None:
            column_index = range(0, results.shape[1])
        tbl = Table(table, self._folder, verbose=verbose)
        if tbl.initialised and shape is None and indices is None:
            tbl.extend(results, row_index)
        elif tbl.initialised:
            tbl[indices] = results.astype(dtype) if dtype is not None else results
        else:
            tbl.create(results, row_index, column_index, shape, indices, dtype)
        return tbl
