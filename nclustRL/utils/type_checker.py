from pathlib import Path

from nclustenv.version import ENV_LIST

from errors import TrainerError, EnvError
from ray.rllib.agents.trainer import Trainer


def is_trainer(trainer):

    if not isinstance(trainer, Trainer):
        raise TrainerError
    return trainer


def is_env(env):

    if env not in ENV_LIST:
        raise EnvError
    return env


def is_dir(dir):

    if not Path(dir).is_dir():
        raise NotADirectoryError('{} directory does not exist'.format(dir))
    return dir


def is_config(config):

    if not isinstance(config, dict):
        raise AttributeError('config parameter should be a dict')
    return config
