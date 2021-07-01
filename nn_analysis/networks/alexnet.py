import numpy as np

from .network import Network
try:
    import torch
    no_torch = False
except ImportError:
    torch = None
    no_torch = True

try:
    from torchvision.models import alexnet
    from torchvision.models import AlexNet as AlexNetModel
    no_torchvision = False
except ImportError:
    AlexNetModel = None
    alexnet = None
    no_torchvision = True


class AlexNet(Network):
    """
    `Network` that accesses the AlexNet models, presenting it stimuli and returning the activations in an interpretable way.

    Attributes:
        model: The actual AlexNet model
        labels: The network labels. This attribute can be used as a names input for storing results.
        input_shape: The shape of the input that AlexNet requires
    """

    def __init__(self):
        if no_torch and no_torchvision:
            raise ImportError('This package requires torch and torchvision to be installed. Please make sure these packages are correctly installed.')
        if no_torch:
            raise ImportError('This package requires the torch package to be installed.')
        if no_torchvision:
            raise ImportError('This package requires the torchvision package to be installed.')
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
                self.__raw_output[name] = output.cpu().detach().numpy()

            return hook
        for submodel_name, submodel in self.model.named_modules():
            if type(submodel) is not AlexNetModel and type(submodel) is not torch.nn.Sequential:
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

        # Move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.to("cuda")

        # Run the model without gradients (since we're not training and the network is not recurrent)
        with torch.no_grad():
            self.model.float()(input_tensor.float())

        return list(self.__raw_output.values()), self.labels
