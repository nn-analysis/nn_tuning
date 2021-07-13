"""
.. include:: ./introduction.md
.. include:: ./getting_started.md
.. include:: ./doc_neural_network.md
.. include:: ./doc_stimulus_set.md
.. include:: ./doc_module_diagram.md
"""
__docformat__ = "google"
__version__ = "1.0.1"

from .storage import *
from .fitting_manager import FittingManager
from .input_manager import *
from .output_manager import *

from .networks import *
from .stimulus_generator import *
from .statistics_helper import *
