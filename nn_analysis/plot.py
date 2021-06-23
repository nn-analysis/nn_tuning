# import mpl_scatter_density
try:
    import matplotlib.pyplot as plt
    no_plotting = False
except ImportError:
    no_plotting = True
from datetime import datetime


class Plot:

    save_fig = False
    save_fig_folder = './'
    filetype = 'png'
    title = None

    @staticmethod
    def show(plot):
        if no_plotting:
            raise ImportError('This function requires matplotlib. Please install matplotlib and try again.')
        plt.tight_layout(0.5)
        if Plot.save_fig:
            tmp_title = Plot.title if Plot.title is not None else datetime.now().microsecond
            plt.savefig(f'{Plot.save_fig_folder}{tmp_title}.{Plot.filetype}')
        else:
            plot.show()
        plot.close()
        plot.clf()
        Plot.title = ''
