import numpy as np
from tqdm import tqdm

from .storage_manager import StorageManager


class FittingManager:

    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager

    @staticmethod
    def get_identity_stim_variables(size_x, size_y) -> (np.array, np.array):
        stim_x = []
        stim_y = []
        for x in range(1, size_x + 1):
            for y in range(1, size_y + 1):
                stim_x.append(x)
                stim_y.append(y)
        return np.array(stim_x), np.array(stim_y)

    def calculate_best_fits(self, results: Union[Table, np.array], table: str = None):
        best_predicted = np.zeros((results.shape[1], 4))
        for i in range(results.shape[1]):
            best_r2 = np.nanmax(results, axis=0)[i]
            best_index = np.where(results[:, i] == best_r2)[0]
            best_x, best_y, best_s = p[best_index, 0][0], p[best_index, 1][0], p[best_index, 2][0]
            best_predicted[i] = best_r2, best_x, best_y, best_s
            i += 1
        if table is not None:
            self.__save__(table, best_predicted, p, override, indices_slice, ncols, verbose, columns,
                          dtype)
        return best_predicted

    def fit_response_function(self, responses: np.ndarray, stim_x: np.array, stim_y: np.array, candidate_function_parameters,
                              prediction_function: str = "np.exp(((stim_x - x) ** 2 + (stim_y - y) ** 2) / (-2 * s ** 2))",
                              stimulus: np.ndarray = None,
                              parallel: bool = None, verbose: bool = False, table: str = None,
                              override: bool = False, indices_slice: slice = None, ncols: int = None,
                              columns: list = None, dtype: np.dtype = None) -> (np.array, np.array):
        if parallel is None:
            parallel = responses.ndim == 2
        if responses.ndim == 1:
            responses = responses[np.newaxis, ...]

        if stimulus is None:
            stimulus = np.eye(len(stim_x))

        if parallel:
            var_resp = np.var(responses, axis=1)
            o = np.ones((responses.shape[1], 1))

        responses_T = responses.T

        r2_result = np.zeros((parameter_set.shape[0], responses.shape[0]), dtype=dtype)
        for row in tqdm(range(0, parameter_set.shape[0]), disable=(not verbose)):
            x, y, s = parameter_set[row]
            g = eval(prediction_function)  # 20480,
            pred = (stimulus @ g)[..., np.newaxis]  # 20480,
            if parallel:
                # noinspection PyUnboundLocalVariable
                _x = np.concatenate((pred, o), axis=1)
                scale = np.linalg.pinv(_x) @ responses_T
                u = np.var(responses_T - _x @ scale, axis=0)
                # noinspection PyUnboundLocalVariable
                r2 = 1 - (u / var_resp)
                r2[np.isnan(r2)] = 0
                r2[r2 == -np.inf] = 0
                r2_result[row] = r2
            else:
                for response in range(0, responses.shape[0]):
                    r2 = np.corrcoef(pred.reshape(-1), responses[response])[0, 1]
                    r2_result[row, response] = r2
        if table is not None:
            self.__save__(table, r2_result, parameter_set, override, indices_slice, ncols, verbose, columns, dtype)
        return parameter_set, r2_result

    def __save__(self, table: str, results: np.array, p: np.array, override: bool,
                 indices_slice: slice, ncols: int, verbose: bool, columns: list = None, dtype: np.dtype = None):
        if table is None:
            return
        if override:
            self.storage_manager.remove_table(table)
        ncols = results.shape[1] if ncols is None else ncols
        columns = list(range(0, ncols)) if columns is None else columns
        rows = []
        for x, y, s in p:
            rows.append((x, y, s))
        shape = (len(rows), ncols)
        self.storage_manager.save_results(table, results, rows, columns, shape, (slice(None), indices_slice),
                                          verbose=verbose, dtype=dtype)

    @staticmethod
    def generate_fake_responses(variables, stim_x, stim_y, stimulus) -> np.array:
        nd_list = list()
        for required_x, required_y, _required_s in variables:
            g = np.exp(((stim_x - required_x) ** 2 + (stim_y - required_y) ** 2) / (-2 * _required_s ** 2))
            pred = (stimulus @ g)
            nd_list.append(pred)
        return np.array(nd_list)

    def test_response_fitting(self, variables_to_discover, shape, stimulus, stim_x, stim_y, step, parallel=False,
                              gpu=False, verbose=False) -> (bool, np.array, np.array):

        generated_responses = self.generate_fake_responses(variables_to_discover, stim_x, stim_y, stimulus)
        if gpu and not parallel:
            raise AssertionError("The GPU fit function only supports parallel computing")
        p, result = self.fit_response_function(generated_responses, stim_x, stim_y, shape, step, parallel=parallel,
                                               verbose=verbose)
        return predicted[:, 1:]

    @staticmethod
    def linearise_sigma(log_sigma, preferred_numerosity):
        log_pref_numerosity = np.log(preferred_numerosity)
        # log_sigma = np.log(sigma)
        fwhm_log = log_sigma * (2 * np.sqrt(2 * np.log(2)))
        fwhm_lin = np.exp(log_pref_numerosity + fwhm_log / 2) - np.exp(log_pref_numerosity - fwhm_log / 2)
        return fwhm_lin

    @staticmethod
    def init_parameter_set(step: (float, float, float), shape: (int, int, int), linearise_s: bool = False,
                           log: bool = False):
        i = 0
        if step[2] < 1:
            start_at = int(1/step[2])
        else:
            start_at = 1
        p = np.zeros((int((int((shape[0]/step[0])) * int((shape[1])/step[1])) * int(((shape[2] - 1)/step[2]))), 3),
                     dtype=np.float32)
        for x in range(0, int(shape[0] * (1/step[0]))):
            for y in range(0, int(shape[1] * (1/step[1]))):
                for s in range(start_at, int(shape[2] * (1/step[2]))):
                    if log:
                        p[i] = np.array([np.log(x * step[0]) if x > 0 else 0,
                                         np.log(y * step[1]) if y > 0 else 0, s * step[2]])
                    else:
                        p[i] = np.array([x * step[0], y * step[1], s * step[2]])
                    i += 1
        if linearise_s:
            p[:, 2] = FittingManager.linearise_sigma(p[:, 2], p[:, 0])
        return p
