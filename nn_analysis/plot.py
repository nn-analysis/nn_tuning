# import mpl_scatter_density
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import List, Union, Tuple

from matplotlib.colors import LinearSegmentedColormap

from .storage import Table
from datetime import datetime


class Plot:

    save_fig = False
    save_fig_folder = './'
    filetype = 'png'
    mem_lim_gb = 30
    title = None

    @staticmethod
    def nn_filter(units: np.ndarray):
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
    def feature_map(feature_map: np.ndarray, ncols: int = 6):
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
    def estimated_parameters(results: Table, p: np.ndarray, nodes, col: int, xlabel: str, ylabel: str, colour_label: str, title: str, shape: (int, int), threshold: float = 0):
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
        plot_array[np.where(best_fits < threshold)] = 0
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
    def scatter_plot_network(predicted_fits: Union[np.ndarray, Table], p: np.ndarray, col: int,
                             nodes: List[Union[slice, Tuple[slice, slice]]], threshold: float, ranges: Tuple[Tuple[int, int], Tuple[int, int]],
                             titles: List[str], y_label: str, col_two: int = -1, x_label: str = 'Goodness of Fit',
                             dtypes: Tuple[np.dtype, np.dtype] = (None, None)):
        fig, axs = plt.subplots(int(np.floor(np.sqrt(len(nodes))))+1, int(np.floor(np.sqrt(len(nodes)))))
        fig.set_size_inches(25, 20)
        i = 0
        for node_slice in nodes:
            x, y, x_range, y_range = Plot.scatter_plot_x_y(predicted_fits, p, col, node_slice,
                                                           threshold, ranges, dtypes, col_two)
            ax = axs.flat[i]
            ax.set_ylim(*y_range)
            ax.set_xlim(*x_range)
            ax.hist2d(x, y, density=True)
            ax.set_title(titles[i])
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_label)
            i += 1
        Plot.show(plt)

    @staticmethod
    def scatter_plot_layer(predicted_fits, p, col, node_slice, threshold, col_two, title, y_label, x_label, ranges: Tuple[Tuple[int, int], Tuple[int, int]], dtypes: Tuple[np.dtype, np.dtype] = (None, None)):

        # Calculate points
        x, y, x_range, y_range = Plot.scatter_plot_x_y(predicted_fits, p, col, node_slice, threshold, ranges, dtypes,
                                                       col_two)
        # # Calculate the point density
        # xy = np.vstack([x, y])
        # z = gaussian_kde(xy)(xy)
        # # Sort the points by density, so that the densest points are plotted last
        # idx = z.argsort()
        # x, y, z = x[idx], y[idx], z[idx]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.set_ylim(*y_range)
        ax.set_xlim(*x_range)

        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)
        density = ax.scatter_density(x, y, cmap=white_viridis)

        fig.colorbar(density, label='Number of neurons per pixel')
        # ax.hist2d(x, y, density=True)

        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        Plot.show(plt)

    @staticmethod
    def scatter_plot_x_y(predicted_fits: Union[np.ndarray, Table], p: np.ndarray, col: int,
                         nodes: Union[np.ndarray, slice],
                         threshold: float,
                         ranges: Tuple[Tuple[int, int], Tuple[int, int]],
                         dtypes: Tuple[np.dtype, np.dtype] = (None, None), col_two: int = -1):
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

        max_x = np.max(x)
        min_x = np.min(x)
        max_y = np.max(y)
        min_y = np.min(y)
        if ranges[0] is not None:
            max_y = ranges[0][-1] if ranges[0][-1] is not None else max_y
            min_y = min_y if ranges[0][0] is None else ranges[0][0]
        if ranges[1] is not None:
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

        if dtypes[0] is not None:
            x = x.astype(dtypes[0])
        if dtypes[1] is not None:
            y = y.astype(dtypes[1])

        return x, y, (min_x, max_x), (min_y, max_y)
