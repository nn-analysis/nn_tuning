from typing import Union

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
        presentation: The presentation of the stimuli. "single_pass" by default. Making it "iterative" will present each intermediate frame separately
        time_points_to_measure (optional): List of which time points should be measured
        take_mean_of_measures (optional, default=True): If True, the measured time points will be averaged and only the average will be stored
    """

    feedforward_only: bool

    def __init__(self, json_file, weight_file, presentation, time_points_to_measure=None, take_mean_of_measures=True):
        if not self.is_tf_one():
            raise AssertionError("Expected module Tensorflow to have version < 2, found tensorflow version " +
                                 tf.__version__)
        self.feedforward_only = False
        if presentation is not "iterative" and presentation is not "single_pass":
            raise ValueError(f'Presentation should be "iterative" or "single_pass", found "{presentation}"')
        self.presentation = presentation
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
        ntime = input_array.shape[1]
        batch_outputs = self.__init_result_array(ntime)
        prednet = self._test_model.layers[1]
        prednet.feedforward_only = self.feedforward_only
        # for s in range(input_array.shape[0]):
        if self.presentation is 'iterative':
            iterator = self.time_points_to_measure if self.time_points_to_measure is not None else range(1, input_array.shape[1]+1)
            for j in iterator:
                raw_step_outputs = self.__call_prednet(prednet, input_array[:, 0:j].reshape((input_array.shape[0], len(list(range(0, j))), input_array.shape[2], input_array.shape[3], input_array.shape[4])).astype(np.float32))
                batch_outputs = self.__extract_from_step_output(raw_step_outputs, batch_outputs, j-1)
        elif self.presentation is 'single_pass':
            batch_input = input_array[:]
            batch_input_tensor = tf.convert_to_tensor(batch_input, np.float32)
            raw_step_outputs = self.__call_prednet(prednet, batch_input_tensor)
            batch_outputs = self.__extract_from_step_output(raw_step_outputs, batch_outputs, 0)
        else:
            raise ValueError(f'Expected presentation to be either iterative or single_pass, found {self.presentation}')
        if len(batch_outputs) == 1:
            return batch_outputs[0]
        return batch_outputs

    @staticmethod
    def __call_prednet(prednet, stimulus) -> dict:
        # initialise an output structure for all the results based on the layer and subtype
        #   Layer type --> Layer
        outputs = {'R': [], 'Ahat': [], 'A': [], 'E': []}
        # go through a list of output types
        for layer_type, _ in outputs.items():
            for layer in range(0, 4):
                # for each item, change the output type of prednet
                output_mode = f'{layer_type}{layer}'
                prednet.output_mode = output_mode
                prednet.output_layer_type = output_mode[:-1]
                prednet.output_layer_num = int(output_mode[-1])
                # run prednet
                outputs[layer_type].append(prednet.call(stimulus))
        # after the loop return the result
        outputs = {'R': outputs['R'], 'Â': outputs['Ahat'], 'A': outputs['A'], 'E': outputs['E']}
        return outputs

    def __init_result_array(self, number_of_timesteps):
        """
        Initialises a result array based on the number of time steps in the input video

        Args:
            number_of_timesteps: The number of time steps in the video (the second dimension in the input_array).

        Returns:
            A list with the network structure for each time step.
        """
        final = []
        if self.feedforward_only:
            base = {
                'A': [[], [], [], []],
                'E': [[], [], [], []]
            }
        else:
            base = {
                'R': [[], [], [], []],
                'Â': [[], [], [], []],
                'A': [[], [], [], []],
                'E': [[], [], [], []]
            }
        for i in range(number_of_timesteps):
            final.append(base)
        return final

    @staticmethod
    def __extract_from_step_output(step_output, final_output, time_step):
        for layer_type, layers in step_output.items():
            for layer in range(len(layers)):
                final_output[time_step][layer_type][layer] = layers[layer]
        return final_output

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
                names[f'{i+1}'][layer_type] = layer_type
                results[i].append(layer)
                i += 1
        return self.__list_to_tuple_recursively(results), names

    def __list_to_tuple_recursively(self, input_list) -> tuple:
        """
        Transforms a list into a tuple recursively.

        Args:
            input_list: List to be transformed

        Returns:
            The resulting tuple
        """
        if not (type(input_list) is list or type(input_list) is tuple):
            return input_list
        for i in range(len(input_list)):
            input_list[i] = self.__list_to_tuple_recursively(input_list[i])
        return tuple(input_list)

    def __calculate_mean(self, outputs: Union[tuple, list]) -> dict:
        """
        Calculates the mean from output with multiple time steps in the iterative recording case.
        This is useful in old versions of tensorflow and pytorch where easier functions for this don't exist.

        This function assumes the first dimension in the output tuple or list is the time dimension.
        The substructure can be a nested structure of dictionaries, tuples or lists.

        The output of the function will have the same dimensions and types as the substructure of the outputs argument.

        Args:
            outputs: The output that has multiple steps to take the mean over

        Returns:
            A dict with just the mean output
        """
        # Define a sum function for Union[dict, list, tuple]
        def sum_structure(x: Union[dict, list, tuple, np.ndarray], y: Union[dict, list, tuple]) \
                -> Union[dict, list, tuple, np.ndarray]:
            # use x as output structure, x is destroyed
            # check the type of structure
            type_x = type(x)
            # Return if this is a leaf
            if type_x is np.ndarray:
                return x + y
            # Otherwise, fill in the x variable according to the type it was before
            if type_x is dict:
                for key, value in x.items():
                    x[key] = sum_structure(x[key], y[key])
            if type_x is list:
                for i in range(len(x)):
                    x[i] = sum_structure(x[i], y[i])
            if type_x is tuple:
                # Make a list, tuples cannot be changed
                new_x = []
                for i in range(len(x)):
                    new_x.append(sum_structure(x[i], y[i]))
                return new_x
            return x

        # Define a function to divide values of a nested structure by a float
        def divide_structure_by(x: Union[dict, list, tuple, np.ndarray], division: float) \
                -> Union[dict, list, tuple, np.ndarray]:

            # use x as output structure, x is destroyed
            # check the type of structure
            type_x = type(x)
            # Return if this is a leaf
            if type_x is np.ndarray:
                return x / division
            # Otherwise, fill in the x variable according to the type it was before
            if type_x is dict:
                for key, value in x.items():
                    x[key] = divide_structure_by(x[key], division)
            if type_x is list:
                for i in range(len(x)):
                    x[i] = divide_structure_by(x[i], division)
            if type_x is tuple:
                # Make a list, tuples cannot be changed
                new_x = []
                for i in range(len(x)):
                    new_x.append(divide_structure_by(x[i], division))
                return new_x
            return x

        # Loop over highest dimension
        summed_output = self.extract_numpy_array(outputs[0])
        for output in outputs:
            # Use the sum function to sum the dimensions per two, first extracting values from the structure.
            summed_output = sum_structure(summed_output, self.extract_numpy_array(output))
        # Divide the resulting sum by the length of the first dimension
        mean_output = divide_structure_by(summed_output, len(outputs))
        return mean_output

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
            else:
                full_names = dict()
                for i in range(len(reshaped_output)):
                    full_names[f'iteration.{i}'] = names
                return tuple(self.extract_numpy_array(reshaped_output)), full_names
        return self.extract_numpy_array(self.__reshape_batch_output(output))
