import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import lognorm, norm
from typing import List, Union, Tuple

from code_analysis import Table
from datetime import datetime


class Plot:

    save_fig = False
    save_fig_folder = './'
    filetype = 'png'
    mem_lim_gb = 30
    title = None

    @staticmethod
    def nn_filter(units: np.array):
        filters = units.shape[3]
        plt.figure(1, figsize=(20, 20))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[:, :, 0, i], interpolation="nearest", cmap="gray")
        Plot.show(plt)

    @staticmethod
    def show(plot):
        plt.tight_layout(0.5)
        if Plot.save_fig:
            tmp_title = Plot.title if Plot.title is not None else datetime.now().microsecond
            plt.savefig(f'{Plot.save_fig_folder}{tmp_title}.{Plot.filetype}')
        else:
            plot.show()
        plot.close()
        plot.clf()
        Plot.title = ''

    @staticmethod
    def feature_map(feature_map: np.array, ncols: int = 6):
        feature_maps = feature_map.shape[1]
        print(np.sum(feature_map))
        plt.figure(1, figsize=(20, 20))
        nrows = math.ceil(int(feature_maps) / ncols) + 1
        for i in range(feature_maps):
            plt.subplot(nrows, ncols, i + 1)
            plt.title('Filter ' + str(i))
            plt.imshow(feature_map[0, i, :, :],
                       interpolation="nearest",
                       cmap="gray")
        Plot.show(plt)

    @staticmethod
    def heatmap(p: np.array, fitting_result: np.array, static_column: int, static_column_value: float, x_column: int,
                y_column: int, node: int, xlabel: str, ylabel: str, colour_label, title: str,
                x_range: (int, int), y_range: (int, int)):
        interesting_results = fitting_result[node, np.where(p[:, static_column] == static_column_value)]
        interesting_results = interesting_results.reshape(-1)
        interesting_p = p[np.where(p[:, static_column] == static_column_value)]
        max_x = int(interesting_p[-1, x_column])
        max_y = int(interesting_p[-1, y_column])
        plot_array = np.zeros((max_x, max_y))
        unique_y = np.unique(interesting_p[:, y_column])
        unique_x = np.unique(interesting_p[:, x_column])
        for i in range(0, interesting_results.shape[0]):
            xi = np.where(unique_x == interesting_p[i, x_column])
            yi = np.where(unique_y == interesting_p[i, y_column])
            plot_array[xi, yi] = interesting_results[i]
        plot_array = plot_array[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        image = ax.imshow(plot_array, cmap="viridis")
        cb = plt.colorbar(image, ax=ax)
        cb.set_label(colour_label)
        Plot.show(plt)

    @staticmethod
    def preferred_xys_distribution(results: Table, p: np.array, nodes, col: int, xlabel: str, ylabel: str, colour_label: str, title: str, shape: (int, int), threshold: float = 0):
        best_fit_tbl = results.calculate_best_fits(p)
        if isinstance(nodes, np.ndarray) and nodes.dtype is np.dtype('O'):
            nodes = np.concatenate(nodes)
        best_fits, pref_x, pref_y, pref_s = best_fit_tbl[:, nodes]
        plot_array = np.zeros(best_fits.shape[0])
        if col == 0:
            plot_array = pref_x
        elif col == 1:
            plot_array = pref_y
        elif col == 2:
            plot_array = pref_s
        # plot_array[np.where(best_fits < threshold)] = 0
        plot_array = plot_array.reshape(shape)
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        image = ax.imshow(plot_array, cmap="viridis", origin='lower')
        cb = plt.colorbar(image, ax=ax)
        cb.set_label(colour_label)
        Plot.show(plt)

    @staticmethod
    def goodness_of_fit(results: np.array, xlabel: str, ylabel: str, colour_label: str, title: str, shape: (int, int)):
        best_fits = np.amax(results, axis=1)
        plot_array = best_fits.reshape(shape)
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        image = ax.imshow(plot_array, cmap="viridis")
        cb = plt.colorbar(image, ax=ax)
        cb.set_label(colour_label)
        Plot.show(plt)

    @staticmethod
    def best_fits(predicted_fits: np.array, p: np.array, axis: int = 0):
        argmax = np.nanargmax(predicted_fits, axis)
        rows = []
        for x in range(argmax.shape[0]):
            if axis == 0:
                rows.append([p[x], predicted_fits[argmax[x], x]])
            elif axis == 1:
                rows.append([predicted_fits[x, argmax[x]]])
        return np.array(rows), argmax

    @staticmethod
    def response_function(x: int, sigma: int, max_x: int, log: bool, title: str, x_label: str,
                          actual_values: List[np.array] = None, number_of_repetitions: List[int] = None, stim_x: np.array = None,
                          stim_y: np.array = None, y: int = 0, stimuli: List[np.array] = None):
        def lognorm_params(mode, stddev):
            """
            Given the mode and std. dev. of the log-normal distribution, this function
            returns the shape and scale parameters for scipy's parameterization of the
            distribution.
            """
            p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
            r = p.roots
            sol = r[(r.imag == 0) & (r.real > 0)].real
            shape = np.sqrt(np.log(sol))
            scale = mode * sol
            return shape[0], scale[0]
        if log:
            shape, scale = lognorm_params(x, sigma)
            dist = lognorm(s=shape, scale=scale)
            # dist = lognorm(s=sigma, scale=np.exp(x))
            _x = np.linspace(0, max_x, 100)
        else:
            dist = norm(x, sigma)
            _x = np.linspace(0, max_x, 100)
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_ylabel('Response amplitude')
        ax.set_xlabel(x_label)
        multiplier = 1 / dist.pdf(x)

        # if actual_values is not None:
        #     # Check input consistency
        #     if stimulus is None:
        #         raise ValueError("expected stimulus to be of type np.array")
        #     if stim_y is None:
        #         stim_y = np.zeros(stim_x.shape)
        #
        #     # Calculate g
        #     g = np.exp(((stim_x - x) ** 2) / (-2 * sigma ** 2))
        #
        #     # Calculate scaled input
        #     o = np.ones((actual_values.shape[1], 1))
        #     pred = (stimulus @ g)[..., np.newaxis]
        #     _x = np.concatenate((pred, o), axis=1)
        #     scale = np.linalg.pinv(_x) @ actual_values
        #     actual_values = _x @ scale
        #     ax.scatter(np.array([i for i in range(max_x) for _ in range(number_of_repetitions)]), actual_values, alpha=0.2)
        #
        # ax.plot(_x, dist.pdf(_x) * multiplier)
        # plt.show()
        # return plt

    @staticmethod
    def scatter_plot_network(predicted_fits: Union[np.array, Table], p: np.array, col: int,
                             nodes: List[Union[slice, Tuple[slice, slice]]], threshold: float, ranges: Tuple[Tuple[int, int], Tuple[int, int]],
                             titles: List[str], y_label: str, col_two: int = -1, x_label: str = 'Goodness of Fit',
                             dtypes: Tuple[np.dtype, np.dtype] = (None, None)):
        fig, axs = plt.subplots(int(np.floor(np.sqrt(len(nodes))))+1, int(np.floor(np.sqrt(len(nodes)))))
        fig.set_size_inches(25, 20)
        i = 0
        for node_slice in nodes:
            x, y = Plot.scatter_plot_x_y(predicted_fits, p, col, node_slice, threshold, col_two)
            ax = axs.flat[i]
            if dtypes[0] is not None:
                x = x.astype(dtypes[0])
            if dtypes[1] is not None:
                y = y.astype(dtypes[1])
            try:
                max_x = np.max(x)
                min_y = np.min(y)
                min_x = np.min(x)
            except ValueError:
                break
            max_y = np.max(y)

            if ranges[0] is not None:
                ax.set_ylim(*ranges[0])
                max_y = ranges[0][-1] if ranges[0][-1] is not None else max_y
                min_y = min_y if ranges[0][0] is None else ranges[0][0]
            if ranges[1] is not None:
                ax.set_xlim(*ranges[1])
                max_x = ranges[1][-1] if ranges[1][-1] is not None else max_x
                min_x = min_x if ranges[1][0] is None else ranges[1][0]
            y_2 = np.zeros(y.size+1)
            y_2[:-1] = y
            y_2[-1] = max_y
            y = y_2
            x_2 = np.zeros(x.size+1)
            x_2[:-1] = x
            x_2[-1] = max_x
            x = x_2

            # Filter the out of range values
            out_of_range_x_upper_bound = np.where(x > max_x)[0]
            out_of_range_x_lower_bound = np.where(x < min_x)[0]
            out_of_range_y_upper_bound = np.where(y > max_y)[0]
            out_of_range_y_lower_bound = np.where(y < min_y)[0]
            exclude = (out_of_range_x_lower_bound, out_of_range_x_upper_bound, out_of_range_y_upper_bound, out_of_range_y_lower_bound)
            exclude = np.unique(np.concatenate(exclude))
            _y = np.delete(y, exclude)
            y = _y
            _x = np.delete(x, exclude)
            x = _x

            ax.hist2d(x, y, density=True)
            ax.set_title(titles[i])
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_label)
            i += 1
        Plot.show(plt)

    @staticmethod
    def scatter_plot_layer(predicted_fits, p, col, node_slice, threshold, col_two, title, y_label, x_label, ranges: Tuple[Tuple[int, int], Tuple[int, int]], dtypes: Tuple[np.dtype, np.dtype] = (None, None)):
        fig, ax = plt.subplots()
        x, y = Plot.scatter_plot_x_y(predicted_fits, p, col, node_slice, threshold, col_two)
        if dtypes[0] is not None:
            x = x.astype(dtypes[0])
        if dtypes[1] is not None:
            y = y.astype(dtypes[1])
        max_x = np.max(x)
        min_x = np.min(x)
        max_y = np.max(y)
        min_y = np.min(y)
        if ranges[0] is not None:
            ax.set_ylim(*ranges[0])
            max_y = ranges[0][-1] if ranges[0][-1] is not None else max_y
            min_y = min_y if ranges[0][0] is None else ranges[0][0]
        if ranges[1] is not None:
            ax.set_xlim(*ranges[1])
            max_x = ranges[1][-1] if ranges[1][-1] is not None else max_x
            min_x = min_x if ranges[1][0] is None else ranges[1][0]

        y_2 = np.zeros(y.size+1)
        y_2[:-1] = y
        y_2[-1] = max_y
        y = y_2
        x_2 = np.zeros(x.size+1)
        x_2[:-1] = x
        x_2[-1] = max_x
        x = x_2

        # Filter the out of range values
        out_of_range_x_upper_bound = np.where(x > max_x)[0]
        out_of_range_x_lower_bound = np.where(x < min_x)[0]
        out_of_range_y_upper_bound = np.where(y > max_y)[0]
        out_of_range_y_lower_bound = np.where(y < min_y)[0]
        exclude = (out_of_range_x_lower_bound, out_of_range_x_upper_bound, out_of_range_y_upper_bound, out_of_range_y_lower_bound)
        exclude = np.unique(np.concatenate(exclude))
        _y = np.delete(y, exclude)
        y = _y
        _x = np.delete(x, exclude)
        x = _x

        ax.hist2d(x, y, density=True)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        Plot.show(plt)

    @staticmethod
    def scatter_plot_x_y(predicted_fits: Union[np.array, Table], p: np.array, col: int,
                         nodes: Union[np.ndarray, slice],
                         threshold: float, col_two: int = -1):
        best_fit_tbl = predicted_fits.calculate_best_fits(p)
        if isinstance(nodes, np.ndarray) and nodes.dtype is np.dtype('O'):
            nodes = np.concatenate(nodes)
        goodness_of_fits, pref_x, pref_y, pref_s = best_fit_tbl[:, nodes]
        if col_two is -1:
            x = goodness_of_fits
        elif col_two is -2:
            x = np.sqrt(np.square(pref_x-80)+np.square(pref_y-64))
        elif col_two is 0:
            x = pref_x
        elif col_two is 1:
            x = pref_y
        else:
            x = pref_s
        if col is -1:
            y = goodness_of_fits
        elif col is -2:
            y = np.sqrt(np.square(pref_x-80)+np.square(pref_y-64))
        elif col is 0:
            y = pref_x
        elif col is 1:
            y = pref_y
        else:
            y = pref_s
        y = y[np.where(goodness_of_fits > threshold)]
        x = x[np.where(goodness_of_fits > threshold)]
        return x, y

    @staticmethod
    def scatter_plot(predicted_fits: Union[np.array, Table], p: np.array, col: int, nodes: Union[slice, Tuple[slice, slice]],
                     threshold: float, title: str, y_label: str, col_two: int = -1, x_label: str = 'Goodness of fit'):
        x, y = Plot.scatter_plot_x_y(predicted_fits, p, col, nodes, threshold, col_two,)
        fig, ax = plt.subplots()
        ax.hist2d(x, y, density=True)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        Plot.show(plt)

    @staticmethod
    def best_col(predicted_fits: np.array, p: np.array, col: int, nnodes: int):
        cs = np.unique(p[:, col])
        maxs = []
        for c in cs:
            maxs.append(np.nanmax(predicted_fits[:, np.where(p[:, col] == c)]))
        maxs = np.array(maxs)
        max_max = np.max(maxs)
        max_col = cs[np.where(maxs == max_max)]

        sorted_list = -np.sort(-predicted_fits[:, np.where(p[:, col] == max_col)].reshape(-1))
        max_col_nodes = []
        for r in sorted_list:
            node = np.where(predicted_fits == r)[0]
            if node not in max_col_nodes:
                max_col_nodes.append(node)
            if len(max_col_nodes) == nnodes:
                break

        return max_col[0], max_col_nodes, maxs

class PlotCache:

    def __init__(self, pref_x, pref_y):
        pass