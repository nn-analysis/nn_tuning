Neural Network Tuning Analysis Toolkit
====

Analyse neural networks for feature tuning.



[Documentation]

[Documentation]: https://nn-analysis.github.io/nn_tuning/nn_tuning.html

Installation
------------

    $ pip install nn_analysis

Depending on your use you might need to install several other packages.

The AlexNet network requires you to install PyTorch and PyTorch vision using:

    $ pip install torch torchvision

PredNet requires a more specific configuration. For PredNet you need to be using python version 3.6 and TensorFlow version < 2.

Features
--------

* Fitting tuning functions to recorded activations of a neural network,
* Automatic storage of large tables on disk in understandable folder structures,
* Easily [extendable] to other neural networks and stimuli.

[extendable]: https://nn-analysis.github.io/nn_tuning/nn_tuning.html#adding-new-neural-networks-to-the-code-analysis-system

The above features are explained in more detail in nn_analyis' [documentation].
