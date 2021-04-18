import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from typing import List

from tqdm import tqdm

from code_analysis import StorageManager, Table, math, Plot
from code_analysis.input_generator import InputGenerator


class NumerosityInputGenerator(InputGenerator):

    def __init__(self, nvars: int, nrange: (int, int), table: str, storage_manager: StorageManager,
                 verbose: bool = False, calc_functions=None):
        if calc_functions is None:
            calc_functions = ['area', 'size', 'circumference']
        self.nvars = nvars
        self.nrange = nrange
        self.__storage_manager = storage_manager
        self.__table = table
        self.__verbose = verbose
        self.__calc_functions = calc_functions
        self.__q = 0

    def _get_2d(self, shape: (int, int), index: int) -> np.array:
        # Only return one colour channel, all colour channels are equal
        return self.generate_dot_image(index, shape, self.__calc_functions[self.__q])[:, :, 0]

    def generate(self, shape: tuple):
        tbl = None
        col_index = Table.shape_to_indices(shape)
        for n in tqdm(range(self.nrange[0], self.nrange[1]), leave=False, disable=(not self.__verbose)):
            for q in range(len(self.__calc_functions)):
                self.__q = q
                for j in range(0, self.nvars):
                    tbl = self.__storage_manager.save_results(self.__table, self._generate_row(shape, n)[np.newaxis, ...],
                                                              [n], col_index)
        return tbl

    @staticmethod
    def generate_dot_image(ndots: int, img_shape: (int, int), calculation_function: str = 'area',
                           plt_image: bool = False, plt_title: str = ''):
        """
        Function to generate numerosity dot images
        @param plt_title: title used in the plot if plt_image is True
        @param plt_image: if True the generated image is plotted
        @param ndots: number of dots to generate
        @param img_shape: Size of the image
        @param calculation_function: Way to calculate dot relation
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

        dotGroup = NumerosityInputGenerator.new_dot_pattern(ndots, width, height, dotSize, recheckDist)
        return NumerosityInputGenerator.draw_image(width, height, dotGroup, dotSize, plt_image, plt_title)

    @staticmethod
    def new_dot_pattern(ndots, width, height, dot_size, recheck_dist):
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
    def draw_image(img_width, image_height, dot_group, dot_size, plt_image: bool = False, plt_title: str = ''):
        """
        Generates the actual Image
        @param plt_title: title used in the plot if plt_image is True
        @param plt_image: if True the generated image is plotted
        @param img_width: width of image
        @param image_height: height of image
        @param dot_group: tuple of x,y coords of circle ([1,39], [43,2], ...)
        @param dot_size: diameter of a single dot
        """

        # make a blank image first
        image = Image.new('RGB', (img_width, image_height))
        draw = ImageDraw.Draw(image)

        r = np.ceil(dot_size / 2)
        for dot in dot_group:
            draw.ellipse((np.ceil(dot[0] - r), np.ceil(dot[1] - r), np.ceil(dot[0] + r), np.ceil(dot[1] + r)), fill=(255, 255, 255))

        image = np.array(image)
        if plt_image:
            plt.imshow(image, origin='lower')
            if Plot.save_fig:
                Plot.title = plt_title
            else:
                plt.title(plt_title)
            Plot.show(plt)
            Plot.title = None

        return image
