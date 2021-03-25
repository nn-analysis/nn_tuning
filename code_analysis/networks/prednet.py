import gc
from typing import Tuple, List

import tensorflow as tf
from keras import Input, Model
from keras.engine.saving import model_from_json
import numpy as np

from code_analysis import Table
from code_analysis.network import Network
from prednet import PredNet


class Prednet(Network):

    _rows: list = None
    _cols: list = None
    feedforward_only: bool = False

    def __init__(self, json_file, weight_file):
        if not self.is_tf_one():
            raise AssertionError("Expected module Tensorflow to have version < 2, found tensorflow version " +
                                 tf.__version__)
        self._train_model, self._test_model = self._setup_prednet_models(json_file, weight_file)

    @staticmethod
    def _setup_pre_trained_model_prednet(output_mode, pre_trained_model, number_of_timesteps):
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

    def run(self, input_array: np.array) -> np.array:
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
        result = np.zeros((input_array.shape[0], 0))
        cols = True
        if self._cols is None:
            self._cols = []
            cols = False
        for layer_type, layers in batch_output.items():
            i = 0
            for layer in layers:
                layer = np.array(layer)
                if not cols:
                    self._cols.extend([layer_type+str(i)+"e"+str(x) for x in list(range(0, layer[0].size))])
                result = np.append(result, layer.reshape((layer.shape[0], -1)), axis=1)
                i += 1
        batch_start = self.current_batch*result.shape[0]
        batch_end = batch_start+result.shape[0]
        self._rows = list(range(batch_start, batch_end))
        return result

    def get_indexes(self) -> (list, list):
        return self._rows, self._cols

    @staticmethod
    def get_result_indices(results: Table, network: str, layer: int, shape: (int, int, int, int),
                           from_index: (int, int, int, int), to_index: (int, int, int, int),
                           from_col: bool = False) -> slice:
        indices = Table.shape_to_indices(shape)
        if not from_col:
            return slice(results.row_from_key(f'{network}{layer}e{indices.index(from_index)}'),
                         results.row_from_key(f'{network}{layer}e{indices.index(to_index)}')+1)
        else:
            return slice(results.col_from_key(f'{network}{layer}e{indices.index(from_index)}'),
                         results.col_from_key(f'{network}{layer}e{indices.index(to_index)}')+1)

    @staticmethod
    def get_network_layer_indices(responses_tbl, feedforward_only: bool = False, layers_to_slice: List[Tuple[str, int, str, int]] = None):
        if layers_to_slice is None:
            layers_to_slice = [('i', 0, 'i', 1), ('i', 1, 'i', 2), ('i', 2, 'i', 3), ('i', 3, 'f', 0), ('f', 0, 'f', 1), ('f', 1, 'f', 2), ('f', 2, 'f', 3),
                               ('f', 3, 'o', 0), ('o', 0, 'o', 1), ('o', 1, 'o', 2), ('o', 2, 'o', 3), ('o', 3, 'c', 0), ('c', 0, 'c', 1), ('c', 1, 'c', 2),
                               ('c', 2, 'c', 3), ('c', 3, '_c', 0), ('_c', 0, '_c', 1), ('_c', 1, '_c', 2), ('_c', 2, '_c', 3), ('_c', 3, '_r', 0), ('_r', 0, '_r', 1),
                               ('_r', 1, '_r', 2), ('_r', 2, '_r', 3), ('_r', 3, 'Ahat', 0), ('Ahat', 0, 'Ahat', 1), ('Ahat', 1, 'Ahat', 2), ('Ahat', 2, 'Ahat', 3),
                               ('Ahat', 3, 'A', 0), ('A', 0, 'A', 1), ('A', 1, 'A', 2), ('A', 2, 'A', 3), ('A', 3, 'R', 0), ('R', 0, 'R', 1), ('R', 1, 'R', 2),
                               ('R', 2, 'R', 3), ('R', 3, 'E', 0), ('E', 0, 'E', 1), ('E', 1, 'E', 2), ('E', 2, 'E', 3), ('E', 3, 'A_before_pool', 0), ('A_before_pool', 0, 'A_before_pool', 1),
                               ('A_before_pool', 1, 'A_before_pool', 2), ('A_before_pool', 2, 'A_before_pool', 3)]
            if feedforward_only:
                layers_to_slice = [('A', 0, 'A', 1), ('A', 1, 'A', 2), ('A', 2, 'A', 3), ('A', 3, 'E', 0), ('E', 0, 'E', 1), ('E', 1, 'E', 2), ('E', 2, 'E', 3), ('E', 3, 'A_before_pool', 0), ('A_before_pool', 0, 'A_before_pool', 1),
                                   ('A_before_pool', 1, 'A_before_pool', 2), ('A_before_pool', 2, 'A_before_pool', 3)]
        slices = []
        for lt_start, l_start, lt_end, l_end in layers_to_slice:
            from_index = responses_tbl.col_from_key(f'{lt_start}{l_start}e{0}')
            try:
                to_index = responses_tbl.col_from_key(f'{lt_end}{l_end}e{0}')
            except ValueError:
                to_index = responses_tbl.shape[1]
            slices.append(slice(from_index, to_index))
        return slices, layers_to_slice

    @staticmethod
    def get_network_combined_layer_indices(responses_tbl, layer: int, feedforward_only: bool = False):
        layers_to_slice = [('i', layer, 'i', layer+1), ('f', layer, 'f', layer+1), ('o', layer, 'o', layer+1),
                           ('c', layer, 'c', layer+1), ('_c', layer, '_c', layer+1), ('_r', layer, '_r', layer+1),
                           ('Ahat', layer, 'Ahat', layer+1), ('A', layer, 'A', layer+1), ('R', layer, 'R', layer+1),
                           ('E', layer, 'E', layer+1)]
        if feedforward_only:
            layers_to_slice = [('A', layer, 'A', layer+1), ('E', layer, 'E', layer+1)]
        if layer < 3 and feedforward_only:
            layers_to_slice.append(('A_before_pool', layer, 'A_before_pool', layer+1))
        if layer and feedforward_only == 3:
            layers_to_slice = [('A', layer, 'E', 0), ('E', layer, 'A_before_pool', 0)]
        slices = []
        for lt_start, l_start, lt_end, l_end in layers_to_slice:
            from_index = responses_tbl.col_from_key(f'{lt_start}{l_start}e{0}')
            try:
                to_index = responses_tbl.col_from_key(f'{lt_end}{l_end}e{0}')
            except ValueError:
                to_index = responses_tbl.shape[1]
            slices.append(np.arange(from_index, to_index))
        return np.array(slices)


