
__author__      = "Pedro Cotovio"
__license__     = 'GNU GPLv3'

from nclustenv.version import VERSION as __version__

import os
import sys
import warnings

from .trainer import Trainer
from . import utils
from . import models
from . import configs
