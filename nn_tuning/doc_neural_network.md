## Adding new neural networks to the code analysis system
In order to extend the code analysis system to new neural networks a new class has to be created for that network in the networks sub package. This class has to extend the network class from that same sub package. 

The new class has to implement a run function. The run function should run a batch of inputs, given in the input variable, through the model, record activations, and return those activations in the form of a nested tuple of np arrays along with a names dictionary that gives names to each of the items in the nested tuple.

Variables specific to the network can be added to the initialisation function of the class.

In practice, for most hierarchical networks, this all means setting up a model in the `__init__` function and running the input through that model in the `run` function. For pre-trained hierarchical PyTorch and TensorFlow/Keras models this means that it is possible to use a fairly standardised approach to building a new network class since recording activations has standardised functions.

### PyTorch models
For PyTorch models it is possible to load the model using the functions in the submodules in `torchvision.models` and then registering hooks for the layers in that model using the following function. 
Hooks are functions that are performed at a certain timepoint in a process. 
In the case of PyTorch, the hooks are called when a layer is called, allowing us to store the intermediate results.
This function also fills a labels variable that you can use as a names variable when returning output in the `run` function. 
Run this function after setting up the model in the `__init__` function. 
For an example of this method in use please see the AlexNet class.

```python
def __register_hooks(self):
    """
    Function that registers hooks to save results from the network model in the run function.
    """
    def hook_wrapper(name: str):
        def hook(_, __, output):
            self.__raw_output[name] = output.cpu().detach().numpy()

        return hook
    for submodel_name, submodel in self.model.named_modules():
        if type(submodel) is not AlexNetModel and type(submodel) is not nn.Sequential:
            self.labels[submodel_name] = submodel_name
            submodel.register_forward_hook(hook_wrapper(submodel_name))
```


Note that this does not work well for recurrent models. 
There you would need to build your own implementation specific to that network.

### TensorFlow models
For TensorFlow you will need a slightly different function but with much of the same idea. 
In the case of TensorFlow, the way to do this generally is to create a second model with the weights of the previous network. 
This new model has layers that are enclosed in a new type of layer that is accessible by hooks in a similar way to PyTorch models. 
The enclosed layer is shown below.

```python
import tensorflow as tf
from typing import List, Callable, Optional

class LayerWithHooks(tf.keras.layers.Layer):
  def __init__(
      self, 
      layer: tf.keras.layers.Layer,
      hooks: List[Callable[[tf.Tensor, tf.Tensor], Optional[tf.Tensor]]] = None):
    super().__init__()
    self._layer = layer
    self._hooks = hooks or []
  
  def call(self, input: tf.Tensor) -> tf.Tensor:
    output = self._layer(input)
    for hook in self._hooks:
      hook_result = hook(input, output)
      if hook_result is not None:
        output = hook_result
    return output
  
  def register_hook(
      self, 
      hook: Callable[[tf.Tensor, tf.Tensor], Optional[tf.Tensor]]) -> None:
    self._hooks.append(hook)
```

The method that registers those hooks is slightly more difficult than the one from PyTorch. An example of a method registering hooks is shown in the code below. In contrast to the default way of doing this in PyTorch, TensorFlow cannot automatically register hooks to each layer. Rather, this code has to be altered based on the network to add each layer in that network to the second model. The second model is the model that should eventually be called in the `run` function.

```python
def __register_hooks(self):
    """
    Function that registers hooks to save results from the network model in the run function.
    """
    self.__raw_output = {}
    def hook_wrapper(name: str):
        def hook(_, output):
            self.__raw_output[name] = tf.identity(output).numpy()

        return hook
    
    # This second model should have the same layers and structure that the original model in self.model has.
    # For each layer you take a copy of the original layer that you add to the new model wrapped in a LayerWithHooks layer.
    # The layer with hooks should have the hook_wrapper that we defined above.
    # The run function should use the self.model2 model to run the network.
    self.model2 = Sequential()
    self.model2.add(LayerWithHooks(Dense(20, 64, weights=model.layers[0].get_weights()), [hook_wrapper('First dense layer')]))
    self.model2.add(Activation('tanh'))
```