from abc import ABC

import numpy as np
import tensorflow as tf


class Network(ABC):

    current_batch: int

    def run(self, input_array: np.array) -> np.array:
        raise NotImplementedError

    def get_indexes(self) -> (list, list):
        raise NotImplementedError

    def is_tf_one(self):
        return tf.__version__ <= "2"

    def extract_numpy_array(self, tensor, session: tf.compat.v1.Session = None):
        if not self.is_tf_one():
            return tensor.numpy()
        if session is not None:
            session.run(tf.compat.v1.global_variables_initializer())
            return session.run(tensor)
        else:
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                return sess.run(tensor)
