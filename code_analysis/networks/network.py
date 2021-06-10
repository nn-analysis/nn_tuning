from abc import ABC

import numpy as np
try:
    import tensorflow as tf
    tensorflow = True
except ImportError:
    tf = None
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

        Args:
            input_array: Input array containing all the stimuli in this batch

        Returns:
            The results as a tuple and the labels as a dictionary
        """
        raise NotImplementedError

    def is_tf_one(self):
        """
        Helper function for checking the TensorFlow function
        """
        if not tensorflow:
            return False
        return tf.__version__ <= "2"

    def extract_numpy_array(self, tensor, session: tf.compat.v1.Session = None):
        """
        If the tensorflow version is version 1, the extraction of arrays from tensors follows a different algorithm.
        This function provides a universal function to perform the operation.
        The session is an optional variable that allows you to share the same session across different extractions
        saving memory.

        Args:
            tensor: The tensor or structure containing tensors that needs to be extracted.
            session (optional): The TensorFlow session.

        Returns:
            The resulting np.ndarray
        """
        if not tensorflow:
            raise ImportError("tensorflow could not be imported")
        if not self.is_tf_one():
            if tf.is_tensor(tensor):
                return tensor.numpy()
            else:
                output = tensor
                tensor_type = type(tensor)
                if tensor_type is dict:
                    for key, value in tensor.items():
                        output[key] = self.extract_numpy_array(tensor[key], session)
                if tensor_type is list:
                    for i in range(len(x)):
                        output[i] = self.extract_numpy_array(tensor[i], session)
                if tensor_type is tuple:
                    # Make a list, tuples cannot be changed
                    new_output = []
                    for i in range(len(tensor)):
                        new_output.append(self.extract_numpy_array(tensor[i], session))
                    return new_output
                return output
        if session is not None:
            session.run(tf.compat.v1.global_variables_initializer())
            return session.run(tensor)
        else:
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                return sess.run(tensor)
