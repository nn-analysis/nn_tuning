from abc import ABC
from typing import Any

import numpy as np

try:
    import tensorflow as tf
    tensorflow = True
except ImportError:
    tf = lambda: None
    tf.compat = lambda: None
    tf.compat.v1 = lambda: None
    tf.compat.v1.Session = Any
    tensorflow = False


class Network(ABC):
    """
    Abstract Network class
    Subclasses of this class will be the only actual interaction point between networks and the rest of the program.

    Attributes:
        current_batch: (int) The current batch
    """

    current_batch: int

    def run(self, input_array: np.ndarray) -> (tuple, dict):
        """
        Runs the stimuli (in the `input_array`) through the network and returns the results.

        Examples
        ---------
        >>> Network().run(np.array([[1,2,3,4], [2,3,4,5]]))
        (tuple of results split up in subparts for the subtable structure of the `TableSet`, dict of names for the structure of the `TableSet`.

        Args:
            input_array: Input array containing all the stimuli in this batch

        Returns:
            The results as a tuple and the labels as a dictionary
        """
        raise NotImplementedError

    @staticmethod
    def is_tf_one():
        """
        Helper function for checking the TensorFlow function
        """
        if not tensorflow:
            return False
        return tf.__version__ <= "2"

    def extract_numpy_array(self, to_extract, session: tf.compat.v1.Session = None):
        """
        If the tensorflow version is version 1, the extraction of arrays from tensors follows a different algorithm.
        This function provides a universal function to perform the operation.

        The session is an optional variable that allows you to share the same session across different extractions
        saving memory.

        Examples
        ---------
        >>> Network().extract_numpy_array([tf.Tensor(), tf.Tensor])
        [Array([]), Array([])]
        >>> Network().extract_numpy_array({'A': tf.Tensor(), 'B': tf.Tensor()})
        {'A': Array([]), 'B': Array([])}

        Args:
            to_extract: The tensor or structure containing tensors that needs to be extracted. This structure can be of any type but may not contain any np.ndarrays.
            session (optional): The TensorFlow session.

        Returns:
            The resulting np.ndarray
        """
        if not tensorflow:
            raise ImportError("tensorflow could not be imported")
        if not self.is_tf_one():
            if tf.is_tensor(to_extract):
                return to_extract.numpy()
            else:
                output = to_extract
                tensor_type = type(to_extract)
                if tensor_type is dict:
                    for key, value in to_extract.items():
                        output[key] = self.extract_numpy_array(to_extract[key], session)
                if tensor_type is list:
                    for i in range(len(to_extract)):
                        output[i] = self.extract_numpy_array(to_extract[i], session)
                if tensor_type is tuple:
                    # Make a list, tuples cannot be changed
                    new_output = []
                    for i in range(len(to_extract)):
                        new_output.append(self.extract_numpy_array(to_extract[i], session))
                    return new_output
                return output
        if session is not None:
            session.run(tf.compat.v1.global_variables_initializer())
            return session.run(to_extract)
        else:
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                return sess.run(to_extract)
