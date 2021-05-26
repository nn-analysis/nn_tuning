import numpy as np

from .error import ShapeMismatchError


def __slicetolist__(key, length: int) -> list:
    """
    Converts slices to lists of indices

    Args:
        key: (slice) The key
        length: (int, optional) The length of the slice. default=nrows

    Returns:
        object: List of indices
    """
    if type(key) is list:
        return key
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


def __keytolist__(key, length: int) -> tuple:
    """
    Converts key to a list as preprocessing for object retrieval/alteration/deletion

    Args:
        key: Key of the index

    Returns:
        object: List of indices
    """
    if type(key) is slice:
        return __slicetolist__(key, length), None
    elif type(key) is tuple:
        if len(key) > 2:
            raise ShapeMismatchError(f"Expected <=2, got {len(key)}")
        rows, cols = key
        if type(rows) is slice:
            rows = __slicetolist__(rows, length)
        elif type(rows) is np.ndarray or type(rows) is list:
            rows = rows
        else:
            rows = [rows]
        return rows, cols
    elif type(key) is np.ndarray or type(key) is list:
        return key, None
    else:
        return [key], None


def __verify_data_types_are_correct__(data: tuple) -> bool:
    """
    Checks whether the data is a tuple of tuples or np.ndarrays everywhere

    Args:
        data: The data of the subtables

    Returns:
        True if it does, False otherwise
    """
    for subdata in data:
        if type(subdata) is tuple:
            if not __verify_data_types_are_correct__(subdata):
                return False
        elif type(subdata) is not np.ndarray and subdata is not None:
            return False
    return True
