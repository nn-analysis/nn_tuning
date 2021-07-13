try:
    import matplotlib.pyplot as plt
    no_plotting = False
except ImportError:
    no_plotting = True
    plt = None
from datetime import datetime

save_fig = False
save_fig_folder = './'
filetype = 'png'
title = None


def show(plot):
    """
    Shows a pyplot plot or saves it depending on the settings

    Args:
        plot: The pyplot object
    """
    if no_plotting:
        raise ImportError('This function requires matplotlib. Please install matplotlib and try again.')
    plt.tight_layout(0.5)
    if save_fig:
        tmp_title = title if title is not None else datetime.now().microsecond
        plt.savefig(f'{save_fig_folder}{tmp_title}.{filetype}')
    else:
        plot.show()
    plot.close()
    plot.clf()
    plot.title = ''
