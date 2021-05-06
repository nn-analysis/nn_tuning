import math

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from .numba_functions import __two_d_var__, __calculate_prediction__, __transpose2d__, __multiply__, __matmultiply__
from .storage_manager import StorageManager

magma_supported = True
try:
    import skcuda.magma as magma
except OSError:
    magma_supported = False


class FittingManager:

    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager

    @staticmethod
    def get_stims(size_x, size_y) -> (np.array, np.array):
        stim_x = []
        stim_y = []
        for x in range(1, size_x + 1):
            for y in range(1, size_y + 1):
                stim_x.append(x)
                stim_y.append(y)
        return np.array(stim_x), np.array(stim_y)

    def fit_response_function(self, responses: np.ndarray, stim_x: np.array, stim_y: np.array, shape: (int, int, int),
                              step: (float, float, float), stimulus: np.ndarray = None,
                              parallel: bool = None, gpu: bool = False, verbose: bool = False, table: str = None,
                              override: bool = False, indices_slice: slice = None, ncols: int = None, log: bool = False,
                              columns: list = None, dtype: np.dtype = None) -> (np.array, np.array):
        if gpu:
            p, result = self.__fit_response_function_gpu__(responses, stim_x, stim_y, shape, step, stimulus)
            if table is not None:
                self.__save__(table, result, p, override, indices_slice, ncols, verbose, columns, dtype)
            return p, result
        if parallel is None:
            parallel = responses.ndim == 2
        if responses.ndim == 1:
            responses = responses[np.newaxis, ...]

        p = self.init_result_array(step, shape, log=log)

        if stimulus is None:
            stimulus = np.eye(len(stim_x))

        if parallel:
            var_resp = np.var(responses, axis=1)
            o = np.ones((responses.shape[1], 1))

        responses_T = responses.T

        r2_result = np.zeros((p.shape[0], responses.shape[0]), dtype=dtype)
        for row in tqdm(range(0, p.shape[0]), disable=(not verbose)):
            x, y, s = p[row]
            g = np.exp(((stim_x - x) ** 2 + (stim_y - y) ** 2) / (-2 * s ** 2))  # 20480,
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
        best_predicted = np.zeros((r2_result.shape[1], 4))
        for i in range(r2_result.shape[1]):
            best_r2 = np.nanmax(r2_result, axis=0)[i]
            best_index = np.where(r2_result[:, i] == best_r2)[0]
            best_x, best_y, best_s = p[best_index, 0][0], p[best_index, 1][0], p[best_index, 2][0]
            best_predicted[i] = best_r2, best_x, best_y, best_s
            i += 1
        if table is not None:
            self.__save__(table, r2_result, p, override, indices_slice, ncols, verbose, columns, dtype)
        if table is not None:
            self.__save__(table+'__best_results', best_predicted, p, override, indices_slice, ncols, verbose, columns, dtype)
        return p, r2_result, best_predicted

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

    def test_response_fitting(self, variables_to_discover, shape, stimulus, stim_x, stim_y, step, parallel=False, gpu=False, verbose=False) -> (
            bool, np.array, np.array):

        generated_responses = self.generate_fake_responses(variables_to_discover, stim_x, stim_y, stimulus)
        if gpu and not parallel:
            raise AssertionError("The GPU fit function only supports parallel computing")
        p, result, predicted = self.fit_response_function(generated_responses, stim_x, stim_y, shape, step,
                                                          parallel=parallel, gpu=gpu, verbose=verbose)
        return predicted[:, 1:]

    @staticmethod
    def __magma_svd__(a, u, s, vt):
        magma.magma_init()
        m, n = a.shape
        optimal_work = np.zeros(1, dtype=np.int)
        iwork = np.zeros(8 * min(m, n), dtype=np.int)
        copy_a = np.array(a)
        magma.magma_sgesdd("S", m, n, copy_a.ctypes.data, m, s.ctypes.data, u.ctypes.data, m, vt.ctypes.data, min(m, n),
                           optimal_work.ctypes.data, -1, iwork.ctypes.data)
        lwork = optimal_work[0]
        work = np.zeros(lwork, dtype=np.int)
        magma.magma_sgesdd("S", m, n, copy_a.ctypes.data, m, s.ctypes.data, u.ctypes.data, m, vt.ctypes.data, min(m, n),
                           work.ctypes.data, lwork, iwork.ctypes.data)
        magma.magma_finalize()

    # @guvectorize([(float32[:, :], float32[:, :], float32[:], float32[:, :], float32[:, :], float32[:, :], float32[:, :],
    #               float32[:, :], float32[:, :])],
    #             '(d,n),(d,d),(d),(d,n),(d,d),(n,d),(d,q),(d,d)->(n,d)', target='cuda')
    # @cuda.jit
    def __pinv__(self, a, u, s, vt, t_u, t_vt, f_s, fs_tu, result):
        # svd(a.ctypes.data, u.ctypes.data, vt.ctypes.data, s.ctypes.data, a.shape[0], a.shape[1], 0)
        self.__magma_svd__(a, u, s, vt)
        cutoff = s[0]
        for i in range(len(s)):
            if cutoff < s[i]:
                cutoff = s[i]
        cutoff *= 1 * math.exp(-15)
        for i in range(len(s)):
            if s[i] > cutoff:
                f_s[i, 0] = 1 / s[i]
            else:
                f_s[i, 0] = 0
        t_u = __transpose2d__(u)
        t_vt = __transpose2d__(vt)
        fs_tu = __multiply__(f_s, t_u)
        return __matmultiply__(t_vt, fs_tu)

    # @guvectorize([(float32[:, :], float32[:], float32[:], float32[:, :], float32[:, :], float32[:],
    #                float32[:], float32[:, :], float32[:, :], float32[:, :], float32[:],
    #                float32[:, :], float32[:, :], float32[:], float32[:, :], float32[:, :], float32[:, :], float32[:, :],
    #                float32[:, :],
    #                float32[:], float32[:], float32[:], float32[:], float32[:, :], float32[:, :])],
    #              '(o, n),(n),(n),(n,n),(a,n),(n),'
    #              '(n),(n,d),(d,a),(n,a),(a),'
    #              '(d,n),(n,n),(n),(n,d),(n,n),(d,n),(n,q),(n,n),'
    #              '(a),(n),(a)->(o, a)',
    #              target='cpu')
    def __fit_response_on_gpu_parallel__(self, p, stim_x, stim_y, stimulus, responses, g,
                                         ones, x, scale, u_n, u,
                                         pinv_x, p_u, s, vt, t_u, t_vt, f_s, fs_tu,
                                         r2, pred, var_resp, result):
        var_resp = __two_d_var__(responses)
        for row in range(0, p.shape[0]):
            print(str(row) + ", " + str(p[row, 0]) + ", " + str(p[row, 1]) + ", " + str(p[row, 2]) + "\r", end="")
            pred = __calculate_prediction__(stim_x, stim_y, p[row, 0], p[row, 1], p[row, 2], __transpose2d__(stimulus),
                                            g)
            transposed_responses = __transpose2d__(responses)
            # Concatenate x a*n
            for i in range(0, 1):
                for _pred in range(0, len(pred)):
                    x[_pred, i] = pred[_pred]
                for _ones in range(0, len(ones)):
                    x[_ones, i + 1] = ones[_ones]

            # pinv_x = np.linalg.pinv(x)
            pinv_x = self.__pinv__(x, p_u, s, vt, t_u, t_vt, f_s, fs_tu, pinv_x)
            scale = __matmultiply__(pinv_x, transposed_responses)  # d*a
            u_n = __matmultiply__(x, scale)  # n * a
            for u_n_row in range(0, u_n.shape[0]):
                for u_n_col in range(0, u_n.shape[1]):
                    u_n[u_n_row, u_n_col] = transposed_responses[u_n_row, u_n_col] - u_n[u_n_row, u_n_col]  # a * n
            u = __two_d_var__(__transpose2d__(u_n))  # a
            for i in range(0, len(r2)):
                r2[i] = 1 - (u[i] / var_resp[i])  # a
            for k in range(0, len(result[row])):
                result[row, k] = r2[k]
        print()

    @staticmethod
    def linearise_sigma(log_sigma, preferred_numerosity):
        log_pref_numerosity = np.log(preferred_numerosity)
        # log_sigma = np.log(sigma)
        fwhm_log = log_sigma * (2 * np.sqrt(2 * np.log(2)))
        fwhm_lin = np.exp(log_pref_numerosity + fwhm_log / 2) - np.exp(log_pref_numerosity - fwhm_log / 2)
        return fwhm_lin

    @staticmethod
    def init_result_array(step: (float, float, float), shape: (int, int, int), linearise_s: bool = False,
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

    def __fit_response_function_gpu__(self, responses: np.ndarray, stim_x: np.array, stim_y: np.array,
                                      shape: (int, int, int), step: (float, float, float),
                                      stimulus: np.ndarray = None) -> (np.array, np.array):
        if not magma_supported:
            raise OSError("MAGMA or CUDA not found. Please make sure CUDA and MAGMA are installed correctly.")
        p = self.init_result_array(step, shape)
        if stimulus is None:
            stimulus = np.eye(len(stim_x))
        n = len(stim_x)
        a = responses.shape[0]
        d = 2
        q = 1
        o = p.shape[0]
        x = np.zeros((n, d), dtype=np.float32)
        scale = np.zeros((d, a), dtype=np.float32)
        u_n = np.zeros((n, a), dtype=np.float32)
        u = np.zeros(a, dtype=np.float32)

        ldu = n
        ldvt = d
        pinv_x = np.zeros((d, n), dtype=np.float32)
        p_u = np.zeros((ldu, min(d, n)), dtype=np.float32)
        s = np.zeros(min(d, n), dtype=np.float32)
        vt = np.zeros((ldvt, d), dtype=np.float32)
        t_u = np.zeros((min(d, n), ldu), dtype=np.float32)
        t_vt = np.zeros((d, ldvt), dtype=np.float32)
        f_s = np.zeros((min(d, n), q), dtype=np.float32)
        fs_tu = np.zeros((n, n), dtype=np.float32)

        r2 = np.zeros(a, dtype=np.float32)
        pred = np.zeros(n, dtype=np.float32)
        var_resp = np.zeros(a, dtype=np.float32)
        ones = np.ones(responses.shape[1], dtype=np.float32)
        result = np.zeros((o, a), dtype=np.float32)
        g = np.zeros(stim_x.shape[0], dtype=np.float32)
        self.__fit_response_on_gpu_parallel__(p, stim_x, stim_y, stimulus, responses, g,
                                              ones, x, scale, u_n, u,
                                              pinv_x, p_u, s, vt, t_u, t_vt, f_s, fs_tu,
                                              r2, pred, var_resp, result)
        return p, result

    def test_all_response_fitting(self, variables_to_discover: list, shape: (int, int), table: str,
                                  verbose: bool = False):
        if verbose:
            print("CPU, Parallel")
        cp_outcome, cp_result_array, p = self.test_response_fitting(variables_to_discover, shape,
                                                                    table=table + "_cpu_parallel",
                                                                    gpu=False, parallel=True)
        if verbose:
            print(cp_outcome)
            print("CPU, Sequential")
        cs_outcome, cs_result_array, _ = self.test_response_fitting(variables_to_discover, shape,
                                                                    table=table + "_cpu_sequential",
                                                                    gpu=False, parallel=False)
        if verbose:
            print(cs_outcome)
            print("GPU, Parallel")
        gp_outcome, gp_result_array, _ = self.test_response_fitting(variables_to_discover, shape,
                                                                    table=table + "_gpu_parallel",
                                                                    gpu=True, parallel=True)
        if verbose:
            print(gp_outcome)
        return (cp_outcome, cs_outcome, gp_outcome), (cp_result_array, cs_result_array, gp_result_array)
