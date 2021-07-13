import random
from enum import Enum
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
    no_plotting = False
except ImportError:
    plt = None
    no_plotting = True
import numpy as np
from PIL import Image, ImageDraw

from tqdm import tqdm
import math
import nn_tuning.plot as plot

from ..storage import StorageManager
from .two_d_stimulus_generator import TwoDStimulusGenerator


class NumerosityStimulusGenerator(TwoDStimulusGenerator):
    """Class containing function pertaining to the generation of stimuli encoding for numerosity

    This class is a subclass of the `TwoDStimulusGenerator` and is responsible for implementing the generation and storage
    of two dimensional numerosity stimuli and the plotting of said stimuli.

    Multiple stimuli are generated for each numerosity for one or multiple different calculation functions.
    The calculation functions are controls for correlating variables with numerosity like the total amount of contrast
    in a stimulus.

    The calculation function can either be area, for a constant total area of all the dots in the image,
    size for a constant dot size, or circumference for a constant total circumference.
    Images are generated with a random dot placement nvar times.

    Attributes:
        nvars: (int) The number of random images that are generated for each numerosity-calculation function pair.
        nrange: (int, int) The range (min, max) of numerosities stimuli will be generated for.
    """

    @property
    def stim_x(self):
        return np.arange(self.nrange[0], self.nrange[1], dtype=np.int)

    @property
    def stim_y(self):
        return np.zeros(self.stim_x.size)

    @property
    def stimulus_description(self) -> np.ndarray:
        """
        Generates the stimulus description for use in the `FittingManager`

        Returns:
            np.ndarray containing the stimulus variable to be used by the FittingManager.
        """
        result = []
        for i in range(*self.nrange):
            result_for_this_numerosity = np.zeros((self.nrange[1]-self.nrange[0]))
            result_for_this_numerosity[i] = 1
            for q in range(len(self.__calc_functions)):
                for j in range(self.nvars):
                    result.append(result_for_this_numerosity)
        return np.array(result)

    def __init__(self, nvars: int, nrange: (int, int), table: str, storage_manager: StorageManager,
                 verbose: bool = False, calc_functions: Optional[List[str]] = None):
        """
        Args:
            nvars: (int) The number of random images that are generated for each numerosity-calculation function pair.
            nrange: (int, int) The range (min, max) of numerosities stimuli will be generated for.
            table: (str) The name of the table the stimuli will be stored to.
            storage_manager: (`StorageManager`) `StorageManager` that will handle the processing of the table
            verbose (optional, default=False): (bool) If yes, will print the process to the console
            calc_functions (optional): List with 'area', 'size', and/or 'circumference.
        """
        if calc_functions is None:
            calc_functions = ['area', 'size', 'circumference']
        elif all(elem != 'area' and elem != 'size' and elem != 'circumference' for elem in calc_functions):
            raise ValueError('Expected calc_functions to only contain \'area\', \'size\', or \'circumference\'')
        self.nvars = nvars
        self.nrange = nrange
        self.__storage_manager = storage_manager
        self.__table = table
        self.__verbose = verbose
        self.__calc_functions = calc_functions
        self.__q = 0

    def _get_2d(self, shape: (int, int), index: int) -> np.ndarray:
        """Generates the 2d stimulus to be appended with other dimensions to a complete stimulus.

        Args:
            shape: (int, int) The shape of the 2d stimulus to generate.
            index: (int) The index of the stimulus. The index allows the function to differentiate which variation to generate in a generalisable way.

        Returns:
            (np.ndarray) The generated stimulus as a 2d image.
        """
        # Only return one colour channel, all colour channels are equal
        return self.__generate_dot_image(index, shape, self.__calc_functions[self.__q])[:, :, 0]

    def generate(self, shape: tuple):
        """Generates the stimuli based on the expected output shape
        It, for each numerosity in `range(self.nrange[0], self.nrange[1)`, for each `self.calc_function`, generates `self.nvar` random placed dot imgaes.
        Finally it saves those images in a `Table` and returns them
        Args:
            shape: (tuple) expected shape of the final stimuli

        Returns:
            (`Table`) The resulting stimuli `Table`
        """
        tbl = None
        for n in tqdm(range(self.nrange[0], self.nrange[1]), leave=False, disable=(not self.__verbose)):
            for q in range(len(self.__calc_functions)):
                self.__q = q
                for j in range(0, self.nvars):
                    tbl = self.__storage_manager.save_result_table_set((self._generate_row(shape, n)[np.newaxis, ...],),
                                                                       self.__table, {self.__table: self.__table}, append_rows=True)
        return tbl

    @staticmethod
    def __generate_dot_image(ndots: int, img_shape: (int, int), calculation_function: str = 'area',
                             plt_image: bool = False, plt_title: str = ''):
        """Function to randomly generate images displaying a set number of dots using a set calculation function.
        The calculation function determines how the function calculates the size of each individual dot.
        The calculation function can either be area, for a constant total area of all the dots in the image, size for a constant dot size, or circumference for a constant total circumference.

        Args:
            ndots: (int) Number of dots to generate
            img_shape: (int, int) Size of the image
            calculation_function (optional, default='area'): (str) Way to calculate dot size
            plt_image (optional, default=False): (bool) If True the generated image is plotted (requires matplotlib to be installed)
            plt_title (optional, default=''): (str) Title used in the plot if plt_image is True

        Returns:
            (np.ndarray) The resulting stimulus
        """
        # Parameters
        win_pos = Enum('win_pos', 'area size circumference')
        ndots = int(ndots)
        width = int(img_shape[1])
        height = int(img_shape[0])
        dotSize = -1
        dotSizeIn = -1
        recheckDist = -1

        # arg1 check
        if ndots < 0:
            raise ValueError("ndots must be >= 0")
        if ndots == 0:
            image = np.zeros((height, width, 3))
            if plt_image:
                plt.imshow(image)
                plt.show()
            return image
        # arg3 check
        if width < 0:
            raise ValueError("shape[0] must be > 0")
        if height < 0:
            raise ValueError("shape[1] must be > 0")
        # arg4 check
        if calculation_function == 'area':
            params_conditionOrder = win_pos.area
        elif calculation_function == 'size':
            params_conditionOrder = win_pos.size
        elif calculation_function == 'circumference':
            params_conditionOrder = win_pos.circumference
        else:
            raise ValueError("calculate_function must be ['area'|'size'|'circumference']")
        # arg compatibility check
        if dotSize >= width or dotSize >= height:
            raise ValueError("dot size cannot be greater or equal to image size in pxs")

        params_equalArea = 1
        # Get Experiment Name
        if params_conditionOrder == win_pos.area:
            params_equalArea = 1
            dotSizeIn = 3 * (7 / 2) ** 2 * math.pi
        elif params_conditionOrder == win_pos.size:
            params_equalArea = 0
            dotSize = 7
        elif params_conditionOrder == win_pos.circumference:
            params_equalArea = 2
            dotSizeIn = 19 * math.pi * 3

        # Get Recheck Distribution Based on ndots
        if params_equalArea == 1:
            dotSize = (2 * (math.sqrt((dotSizeIn / ndots) / math.pi)))
            if ndots == 2:
                recheckDist = 5
            elif ndots == 3:
                recheckDist = 5
            elif ndots == 4:
                recheckDist = 4.8
            elif ndots == 5:
                recheckDist = 4.5
            elif ndots == 6:
                recheckDist = 4.2
            elif ndots == 7:
                recheckDist = 4
            else:
                recheckDist = 3
        elif params_equalArea == 2:
            dotSize = dotSizeIn / ndots / math.pi
            if ndots == 2:
                recheckDist = 1.15
            elif ndots == 3:
                recheckDist = 1.5
            elif ndots == 4:
                recheckDist = 1.9
            elif ndots == 5:
                recheckDist = 2.1
            elif ndots == 6:
                recheckDist = 2.3
            elif ndots == 7:
                recheckDist = 2.5
            else:
                recheckDist = 3
        elif params_equalArea == 0:
            if ndots == 2:
                recheckDist = 6
            elif ndots == 3:
                recheckDist = 5
            elif ndots == 4:
                recheckDist = 4
            elif ndots == 5:
                recheckDist = 3.5
            elif ndots == 6:
                recheckDist = 3
            elif ndots == 7:
                recheckDist = 2.8
            else:
                recheckDist = 1.4

        dotGroup = NumerosityStimulusGenerator.__new_dot_pattern(ndots, width, height, dotSize, recheckDist)
        return NumerosityStimulusGenerator.__draw_image(width, height, dotGroup, dotSize, plt_image, plt_title)

    @staticmethod
    def __new_dot_pattern(ndots: int, width: int, height: int, dot_size: float, recheck_dist):
        """Function that generates the pattern of where the dots have to be placed.

        Args:
            ndots: (int) Number of dots to generate
            width: (int) Width of the stimulus
            height: (int) Height of the stimulus
            dot_size: (float) Size of the dot
            recheck_dist: Variable that determines how close dots are allowed to be

        Returns:
            Tuple of x,y coords of circle ([1,39], [43,2], ...)
        """
        infCounter = 0
        recheckCounter = 1000
        dotGroup = ()
        while recheckCounter == 1000:
            for curdot in range(0, ndots):
                # Dot position, x,y
                # tempDotPattern = [random.uniform(0.1, 0.9) * img_len_px, random.uniform(0.1, 0.9) * img_len_px]
                tempDotPattern = [random.uniform(0.1, 0.9) * width, random.uniform(0.1, 0.9) * height]

                # Get dot pattern that works for img size
                while math.sqrt((tempDotPattern[0] - 0.5 * width) ** 2 + (
                        tempDotPattern[1] - 0.5 * height) ** 2) > 0.5 * math.sqrt(width * height) - dot_size / 2:
                    tempDotPattern = [random.uniform(0.1, 0.9) * width, random.uniform(0.1, 0.9) * height]
                    infCounter = infCounter + 1
                    if infCounter >= 5000:
                        raise ArithmeticError("fatal error....")
                A = tempDotPattern[0]
                B = tempDotPattern[1]

                if curdot == 0:
                    dotGroup = (tempDotPattern,)  # make initial tuple
                    recheckCounter = 1
                else:
                    recheck = 1
                    recheckCounter = 1
                    while recheck == 1:
                        recheck = 0
                        for storedDots in range(0, len(dotGroup)):
                            if recheck == 0:
                                xDist = dotGroup[storedDots][0] - A
                                yDist = dotGroup[storedDots][1] - B
                                totalDist = math.sqrt(xDist ** 2 + yDist ** 2)
                                if totalDist < (dot_size * recheck_dist):
                                    recheck = 1

                        if recheck == 0:
                            dotGroup = (*dotGroup, [A, B])
                        else:
                            tempDotPattern = [random.uniform(0.1, 0.9) * width, random.uniform(0.1, 0.9) * height]
                            while math.sqrt((tempDotPattern[0] - 0.5 * width) ** 2 + (
                                    tempDotPattern[1] - 0.5 * height) ** 2) \
                                    > 0.5 * math.sqrt(width * height) - dot_size / 2:
                                tempDotPattern = [random.uniform(0.1, 0.9) * width,
                                                  random.uniform(0.1, 0.9) * height]
                            A = tempDotPattern[0]
                            B = tempDotPattern[1]
                            recheckCounter = recheckCounter + 1
                            if recheckCounter == 1000:
                                raise ArithmeticError("rechecks exceed maximum, double check parameter tuning")
        return dotGroup

    @staticmethod
    def __draw_image(img_width: int, image_height: int, dot_group: tuple,
                     dot_size: float, plt_image: bool = False, plt_title: str = ''):
        """Generates the actual Image

        Args:
            img_width: (int) Width of image
            image_height: (int) Height of image
            dot_group: (tuple) Tuple of x,y coords of circle ([1,39], [43,2], ...)
            dot_size: (float) Diameter of a single dot.
            plt_image: (bool) If True the generated image is plotted
            plt_title: (str) Title used in the plot if plt_image is True

        Returns:
            np.ndarray containing the image
        """

        # make a blank image first
        image = Image.new('RGB', (img_width, image_height))
        draw = ImageDraw.Draw(image)

        r = np.ceil(dot_size / 2)
        for dot in dot_group:
            draw.ellipse((np.ceil(dot[0] - r), np.ceil(dot[1] - r), np.ceil(dot[0] + r), np.ceil(dot[1] + r)), fill=(255, 255, 255))

        image = np.array(image)
        if plt_image and not no_plotting:
            plt.imshow(image, origin='lower')
            if plot.save_fig:
                plot.title = plt_title
            else:
                plt.title(plt_title)
            plot.show(plt)
            plot.title = None
        elif plt_image and no_plotting:
            print('Matplotlib not found. Skipping image plotting.')

        return image
