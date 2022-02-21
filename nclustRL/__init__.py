
__author__      = "Pedro Cotovio"
__license__     = 'GNU GPLv3'

import os
import sys
import warnings
import ray

# init ray
ray.init(ignore_reinit_error=True)


from .version import VERSION as __version__
from .trainer import Trainer
from . import utils
from . import models
from . import configs

# Import Restart
from .utils.helper import restart_cluster

# register envs in ray
from .utils import registry
from nclustenv.version import ENV_LIST as _ENV_LIST

for env in _ENV_LIST:
    registry.register_env(env)
