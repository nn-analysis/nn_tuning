import gc

import tensorflow as tf
from keras import Input, Model
from keras.engine.saving import model_from_json
import numpy as np

from .network import Network
from prednet import PredNet


class Prednet(Network):
    """
    `Network` that accesses the PredNet models, presenting it stimuli and returning the activations in an interpretable way.

    Attributes:
        feedforward_only: Boolean that determines whether the recurrent connections are used

    Args:
        json_file: JSon file containing the pre-trained model
        weight_file: File containing the pre-trained weights
    """

    feedforward_only: bool

    def __init__(self, json_file, weight_file):
        if not self.is_tf_one():
            raise AssertionError("Expected module Tensorflow to have version < 2, found tensorflow version " +
                                 tf.__version__)
        self.feedforward_only = False
        self._train_model, self._test_model = self._setup_prednet_models(json_file, weight_file)

    @staticmethod
    def _setup_pre_trained_model_prednet(output_mode: str, pre_trained_model, number_of_timesteps):
        """
        Initialises a testing model of PredNet using exiting weights from the training model.

        Args:
            output_mode: The output mode of prednet ['prediction'|'error'|'all'|'R|A|Ahat|E' + layer]
            pre_trained_model: The pre-trained training model.
            number_of_timesteps: The number of timesteps in the input.

        Returns:
            The loaded model
        """
        layer_config = pre_trained_model.layers[1].get_config()
        layer_config['output_mode'] = output_mode  # ['prediction'|'error'|'all'|'R|A|Ahat|E' + layer]
        test_prednet = PredNet(weights=pre_trained_model.layers[1].get_weights(), **layer_config)
        input_shape = list(pre_trained_model.layers[0].batch_input_shape[1:])
        input_shape[0] = number_of_timesteps
        inputs = Input(shape=tuple(input_shape))
        predictions = test_prednet(inputs)
        model = Model(inputs=inputs, outputs=predictions)
        model.layers[1].step_outputs = []
        return model

    def _setup_prednet_models(self, json_file, weights_file):
        """
        Setup the pretrained PredNet test and training models.

        Args:
            json_file: The json file from the pre-trained model.
            weights_file: The weight file from the pre-trained model.

        Returns:
            tuple with the training and testing model
        """
        # Load trained model
        f = open(json_file, 'r')
        json_string = f.read()
        f.close()
        train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
        train_model.load_weights(weights_file)

        # Create testing models
        test_model = self._setup_pre_trained_model_prednet('prediction', train_model, 1)
        test_model.compile(loss='mean_absolute_error', optimizer='adam')
        return train_model, test_model

    def __call_and_run_prednet(self, input_array: np.ndarray) -> dict:
        """
        Actually runs and calls PredNet.
        Returns the raw batch output

        Args:
            input_array: Input array containing all the stimuli in this batch

        Returns:
            Dictionary with all the np.ndarrays from different subparts per layer.
        """
        with tf.compat.v1.Session() as sess:
            batch_output = dict()
            prednet = self._test_model.layers[1]
            prednet.feedforward_only = self.feedforward_only
            for s in range(0, input_array.shape[0]):
                batch_input = input_array[s:s+1]
                batch_input_tensor = tf.convert_to_tensor(batch_input, np.float32)
                prednet.call(batch_input_tensor)
                # Get the outputs from the prednet, making sure to ignore the double values
                outputs: dict = prednet.step_outputs[0]
                for key, value, in outputs.items():
                    if key not in batch_output.keys():
                        batch_output[key] = list()
                    for _l in range(0, len(value)):
                        if not _l < len(batch_output[key]):
                            batch_output[key].append(list())
                        t = value[_l]
                        if not s < len(batch_output[key][_l]):
                            batch_output[key][_l].append(list())
                        batch_output[key][_l][s].append(t)
                prednet.step_outputs = []
            print("Extracting numpy arrays")
            batch_output = self.extract_numpy_array(batch_output, sess)
            sess.close()
        gc.collect()
        return batch_output

    @staticmethod
    def __reshape_batch_output(batch_output: dict) -> (tuple, dict):
        """
        Reshape to desired output shape

        Args:
            batch_output: Raw batch_output dictionary

        Returns:
            The results as a tuple and the labels as a dictionary
        """
        results = []
        names = {}
        for layer_type, layers in batch_output.items():
            i = 0
            for layer in layers:
                if len(results)-1 < i:
                    results.append([])
                    names[f'{i+1}'] = dict()
                layer = np.array(layer).reshape((layer.shape[0], -1))
                names[f'{i+1}'][layer_type] = layer
                i += 1
        return tuple(results), names

    def run(self, input_array: np.ndarray) -> (tuple, dict):
        """
        Runs the stimuli (in the `input_array`) through the network and returns the results.

        Args:
            input_array: Input array containing all the stimuli in this batch

        Returns:
            The results as a tuple and the labels as a dictionary
        """
        batch_output = self.__call_and_run_prednet(input_array)
        return self.__reshape_batch_output(batch_output)
