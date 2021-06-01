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
        time_points_to_measure (optional): List of which time points should be measured
        take_mean_of_measures (optional, default=True): If True, the measured time points will be averaged and only the average will be stored
    """

    feedforward_only: bool

    def __init__(self, json_file, weight_file, time_points_to_measure=None, take_mean_of_measures=True):
        if not self.is_tf_one():
            raise AssertionError("Expected module Tensorflow to have version < 2, found tensorflow version " +
                                 tf.__version__)
        self.feedforward_only = False
        self._train_model, self._test_model = self._setup_prednet_models(json_file, weight_file)
        self.time_points_to_measure = time_points_to_measure
        self.take_mean_of_measures = take_mean_of_measures

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
                if self.time_points_to_measure is not None:
                    step_outputs = [prednet.step_outputs[i] for i in self.time_points_to_measure]
                else:
                    step_outputs = prednet.step_outputs
                batch_outputs = []
                for outputs in step_outputs:
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
                    batch_outputs.append(batch_output)
                prednet.step_outputs = []
            batch_outputs = self.extract_numpy_array(batch_outputs, sess)
            sess.close()
        gc.collect()
        if len(batch_outputs) == 1:
            return batch_outputs[0]
        return batch_outputs

    def __reshape_batch_output(self, batch_output: dict) -> (tuple, dict):
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
                layer = np.array(layer)
                layer = layer.reshape((layer.shape[0], -1))
                names[f'{i+1}'][layer_type] = layer_type
                results[i].append(layer)
                i += 1
        return self.__list_to_tuple_recursively(results), names

    def __list_to_tuple_recursively(self, input_list: list) -> tuple:
        """
        Transforms a list into a tuple recursively.

        Args:
            input_list: List to be transformed

        Returns:
            The resulting tuple
        """

        for i in range(len(input_list)):
            if type(input_list[i]) is list:
                input_list[i] = self.__list_to_tuple_recursively(input_list[i])
        return tuple(input_list)

    @staticmethod
    def __calculate_mean(outputs: list) -> dict:
        """
        Calculates the mean from output with multiple time steps

        Args:
            outputs: The output that has multiple steps to take the mean over

        Returns:
            A dict with just the mean output
        """
        # Create list of indices
        def create_indices_list(recursive_list):
            result = []
            i = 0
            for item in recursive_list:
                if type(item) is list:
                    sublist = create_indices_list(item)
                    for subitem in sublist:
                        result.append(subitem)
                else:
                    result.append(i)
                i += 1
            return result

        # Recursively go through the first output
        indices = create_indices_list(outputs[0])

        # Get and set output from the output list using the an item in the indices list
        def get_item_from_output(single_output, indices_list):
            intermediate_result = single_output
            for inner_index in indices_list:
                intermediate_result = intermediate_result[inner_index]
            return intermediate_result

        def set_item_from_output(single_output, indices_list, value):
            intermediate_results = [single_output]
            for _ in indices_list:
                intermediate_results.append(intermediate_results[-1])
            intermediate_results[-1] = value
            i = len(indices_list)
            result = value
            for intermediate_result in reversed(intermediate_results):
                if i == len(indices_list)-1:
                    _result = intermediate_result
                    _result[indices_list[i]] = result
                    result = _result
                i -= 1
            return result
        # Go through the list of indices
        final_result = outputs[0]
        for index in indices:
            all_arrays = []
            for output in outputs:
                all_arrays.append(get_item_from_output(output, index))
            mean_output = np.mean(np.array(all_arrays), axis=0)
            final_result = set_item_from_output(final_result, index, mean_output)
        return final_result

    def run(self, input_array: np.ndarray) -> (tuple, dict):
        """
        Runs the stimuli (in the `input_array`) through the network and returns the results.

        Args:
            input_array: Input array containing all the stimuli in this batch

        Returns:
            The results as a tuple and the labels as a dictionary
        """
        output = self.__call_and_run_prednet(input_array)
        if type(output) is list:
            names = None
            reshaped_output = []
            for single_time_point_item in output:
                reshaped_output_item, names = self.__reshape_batch_output(single_time_point_item)
                reshaped_output.append(reshaped_output_item)
            if self.take_mean_of_measures:
                reshaped_output = self.__calculate_mean(reshaped_output)
            return reshaped_output, names
        return self.__reshape_batch_output(output)
