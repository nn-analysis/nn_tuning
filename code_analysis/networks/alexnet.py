import numpy as np
from torch import nn

from .network import Network
import torch
from torchvision.models import alexnet
from torchvision.models import AlexNet as AlexNetModel


class AlexNet(Network):
    """
    `Network` that accesses the AlexNet models, presenting it stimuli and returning the activations in an interpretable way.

    Attributes:
        model: The actual AlexNet model
        labels: The network labels. This attribute can be used as a names input for storing results.
        input_shape: The shape of the input that AlexNet requires
    """

    def __init__(self):
        self.model = alexnet(pretrained=True)
        self.labels = {}
        self.__raw_output = {}
        self.input_shape = (3, 256, 256)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.__register_hooks()

    def __register_hooks(self):
        """
        Function that registers hooks to save results from the network model in the run function.
        """
        def hook_wrapper(name: str):
            def hook(_, __, output):
                self.__raw_output[name] = output.detach().numpy()

            return hook
        for submodel_name, submodel in self.model.named_modules():
            if type(submodel) is not AlexNetModel and type(submodel) is not nn.Sequential:
                self.labels[submodel_name] = submodel_name
                submodel.register_forward_hook(hook_wrapper(submodel_name))

    def run(self, input_array: np.ndarray) -> (tuple, dict):
        """
        Runs the stimuli (in the `input_array`) through the network and returns the results.

        Args:
            input_array: Input array containing all the stimuli in this batch

        Returns:
            The results as a tuple and the labels as a dictionary
        """
        self.__raw_output = {}
        input_tensor = torch.from_numpy(input_array)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda")

        with torch.no_grad():
            self.model.float()(input_tensor)

        return list(self.__raw_output), self.labels
