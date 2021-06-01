import numpy as np
from torch import nn

from .network import Network
import torch
from torchvision.models import alexnet
from torchvision.models import AlexNet as AlexNetModel


class AlexNet(Network):
    """
    `Network` that accesses the AlexNet models, presenting it stimuli and returning the activations in an interpretable way.
    """

    def __init__(self):
        self.model = alexnet(True)
        self.labels = {}
        self.__raw_output = {}
        self.input_shape = (3, 256, 256)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.register_hooks()

    def register_hooks(self):
        def hook_wrapper(name: str):
            def hook(_, __, output):
                self.__raw_output[name] = output.detach().numpy()

            return hook
        for submodel_name, submodel in self.model.named_modules():
            if type(submodel) is not AlexNetModel and type(submodel) is not nn.Sequential:
                self.labels[submodel_name] = submodel_name
                submodel.register_forward_hook(hook_wrapper(submodel_name))

    def run(self, input_array: np.ndarray) -> (tuple, dict):
        self.__raw_output = {}
        input_tensor = torch.from_numpy(input_array)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda")

        with torch.no_grad():
            self.model.float()(input_tensor)

        return list(self.__raw_output), self.labels
