from typing import Union

try:
    import tensorflow as tf
    no_tensorflow = False
except ImportError:
    tf = None
    no_tensorflow = True
try:
    from keras import Input, Model
    from keras.engine.saving import model_from_json
    no_keras = False
except ImportError:
    no_keras = True
    Input, Model, model_from_json = None, None, None
import numpy as np

from .network import Network
try:
    from prednet import PredNet
except ImportError:
    PredNet = None


class Prednet(Network):
    """
    `Network` that accesses the PredNet models, presenting it stimuli and returning the activations in an interpretable way.

    Attributes:
        feedforward_only: Boolean that determines whether the recurrent connections are used

    Args:
        json_file: JSon file containing the pre-trained model
        weight_file: File containing the pre-trained weights
        presentation: The presentation of the stimuli. "single_pass" by default. Making it "iterative" will present each intermediate frame separately
        take_mean_of_measures (optional, default=True): If True, the measured time points will be averaged and only the average will be stored
    """

    feedforward_only: bool

    def __init__(self, json_file, weight_file, presentation="single_pass", take_mean_of_measures=True):
        if no_tensorflow:
            raise ImportError("This network requires tensforflow==1.* to be installed")
        if not self.is_tf_one():
            raise AssertionError("Expected module Tensorflow to have version < 2, found tensorflow version " +
                                 tf.__version__)
        self.feedforward_only = False
        if presentation is not "iterative" and presentation is not "single_pass":
            raise ValueError(f'Presentation should be "iterative" or "single_pass", found "{presentation}"')
        self.presentation = presentation
        self._train_model, self._test_model = self._setup_prednet_models(json_file, weight_file)
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
        prednet = self._test_model.layers[1]
        prednet.feedforward_only = self.feedforward_only
        with tf.compat.v1.Session() as sess:
            if self.presentation is 'iterative':
                iterator = range(1, input_array.shape[1]+1)
                combined_batch_output = None
                for j in iterator:
                    # Select the channels
                    reshaped_iterative_input_array = input_array[:, 0:j].reshape((input_array.shape[0], len(list(range(0, j))), input_array.shape[2], input_array.shape[3], input_array.shape[4])).astype(np.float32)
                    # Run PredNet
                    raw_step_outputs = self.__call_prednet(prednet, reshaped_iterative_input_array, True)
                    this_batch_output = self.extract_numpy_array(self.__extract_from_step_output(raw_step_outputs), sess)
                    if combined_batch_output is None:
                        combined_batch_output = this_batch_output
                    else:
                        if self.take_mean_of_measures:
                            combined_batch_output = self.__sum_structure(combined_batch_output, this_batch_output)
                        else:
                            combined_batch_output = self.__concat_in_structure(combined_batch_output, this_batch_output)
                if self.take_mean_of_measures:
                    return self.extract_numpy_array(self.__divide_structure_by(combined_batch_output, input_array.shape[1]))
                else:
                    return self.extract_numpy_array(combined_batch_output, input_array.shape[1])
            elif self.presentation is 'single_pass':
                batch_input = input_array[:]
                batch_input_tensor = tf.convert_to_tensor(batch_input, np.float32)
                raw_step_outputs = self.__call_prednet(prednet, batch_input_tensor)
                if self.take_mean_of_measures:
                    extracted_step_outputs = self.__extract_from_step_output(raw_step_outputs)
                    combined_step_outputs = tf.math.reduce_sum(extracted_step_outputs, 1)
                    return self.extract_numpy_array(self.__divide_structure_by(combined_step_outputs, input_array.shape[1]))
                return self.extract_numpy_array(self.__extract_from_step_output(raw_step_outputs), sess)
            else:
                raise ValueError(f'Expected presentation to be either iterative or single_pass, found {self.presentation}')

    @staticmethod
    def __call_prednet(prednet, stimulus, last_frame_only: bool = False) -> dict:
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
                if last_frame_only:
                    outputs[layer_type].append(prednet.call(tf.convert_to_tensor(stimulus, np.float32))[:, -1])
                else:
                    outputs[layer_type].append(prednet.call(tf.convert_to_tensor(stimulus, np.float32)))
        # after the loop return the result
        outputs = {'R': outputs['R'], 'Â': outputs['Ahat'], 'A': outputs['A'], 'E': outputs['E']}
        return outputs

    def __init_result_array(self):
        """
        Initialises a result array based on the number of time steps in the input video

        Returns:
            A list with the network structure for each time step.
        """
        if self.feedforward_only:
            return {
                'A': [[], [], [], []],
                'E': [[], [], [], []]
            }
        else:
            return {
                'R': [[], [], [], []],
                'Â': [[], [], [], []],
                'A': [[], [], [], []],
                'E': [[], [], [], []]
            }

    def __extract_from_step_output(self, step_output):
        output = self.__init_result_array()
        for layer_type, layers in step_output.items():
            for layer in range(len(layers)):
                output[layer_type][layer] = layers[layer]
        return output

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

    def __sum_structure(self, x: Union[dict, list, tuple, np.ndarray], y: Union[dict, list, tuple, np.ndarray]) \
            -> Union[dict, list, tuple, np.ndarray]:
        """
        Define a sum function for Union[dict, list, tuple]

        Args:
            x: The first input structure
            y: The second input structure

        Returns:
            A summed dict, list, tuple or np.ndarray
        """
        # use x as output structure, x is destroyed
        # check the type of structure
        type_x = type(x)
        # Return if this is a leaf
        if type_x is np.ndarray:
            return x + y
        # Otherwise, fill in the x variable according to the type it was before
        if type_x is dict:
            for key, value in x.items():
                x[key] = self.__sum_structure(x[key], y[key])
        if type_x is list:
            for i in range(len(x)):
                x[i] = self.__sum_structure(x[i], y[i])
        if type_x is tuple:
            # Make a list, tuples cannot be changed
            new_x = []
            for i in range(len(x)):
                new_x.append(self.__sum_structure(x[i], y[i]))
            return new_x
        return x

    def __concat_in_structure(self, x: Union[dict, list, tuple, np.ndarray], y: Union[dict, list, tuple, np.ndarray]) \
                                -> Union[dict, list, tuple, np.ndarray]:
        """
        Define a concatenation function for Union[dict, list, tuple]

        Args:
            x: The first input structure
            y: The second input structure

        Returns:
            A concatenated dict, list, tuple or np.ndarray
        """
        # use x as output structure, x is destroyed
        # check the type of structure
        type_x = type(x)
        # Return if this is a leaf
        if type_x is np.ndarray:
            return tf.concat([x, y], 1)
        # Otherwise, fill in the x variable according to the type it was before
        if type_x is dict:
            for key, value in x.items():
                x[key] = self.__sum_structure(x[key], y[key])
        if type_x is list:
            for i in range(len(x)):
                x[i] = self.__sum_structure(x[i], y[i])
        if type_x is tuple:
            # Make a list, tuples cannot be changed
            new_x = []
            for i in range(len(x)):
                new_x.append(self.__sum_structure(x[i], y[i]))
            return new_x
        return x

    def __divide_structure_by(self, x: Union[dict, list, tuple, np.ndarray], division: float) \
            -> Union[dict, list, tuple, np.ndarray]:
        """
        Define a function to divide values of a nested structure by a float

        Args:
            x: The input structure
            division: The float to divide the items in the structure with

        Returns:
            A divided dict, list, tuple or np.ndarray
        """
        # use x as output structure, x is destroyed
        # check the type of structure
        type_x = type(x)
        # Return if this is a leaf
        if type_x is np.ndarray:
            return x / division
        # Otherwise, fill in the x variable according to the type it was before
        if type_x is dict:
            for key, value in x.items():
                x[key] = self.__divide_structure_by(x[key], division)
        if type_x is list:
            for i in range(len(x)):
                x[i] = self.__divide_structure_by(x[i], division)
        if type_x is tuple:
            # Make a list, tuples cannot be changed
            new_x = []
            for i in range(len(x)):
                new_x.append(self.__divide_structure_by(x[i], division))
            return new_x
        return x

    def run(self, input_array: np.ndarray) -> (tuple, dict):
        """
        Runs the stimuli (in the `input_array`) through the network and returns the results.

        Args:
            input_array: Input array containing all the stimuli in this batch

        Returns:
            The results as a tuple and the labels as a dictionary
        """
        output = self.__call_and_run_prednet(input_array)
        return self.__reshape_batch_output(output)
