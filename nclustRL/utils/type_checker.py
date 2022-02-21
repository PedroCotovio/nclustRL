from pathlib import Path
from dgl import data

from nclustenv.version import ENV_LIST
from nclustenv.utils.datasets import SyntheticDataset

from nclustRL.utils.errors import TrainerError, EnvError, DatasetError
from ray.rllib.agents.trainer import Trainer


def is_trainer(trainer):

    if not isinstance(trainer, type(Trainer)):
        raise TrainerError(trainer)
    return trainer


def is_env(env):

    if env not in ENV_LIST:
        raise EnvError(env)
    return env


def is_dir(dir):

    if not Path(dir).is_dir():
        raise NotADirectoryError('{} directory does not exist'.format(dir))
    return dir


def is_config(config):

    if not isinstance(config, dict):
        raise AttributeError('config parameter should be a dict')
    return config


def is_dataset(dataset):

    if not isinstance(dataset, SyntheticDataset):
        raise DatasetError(dataset)
    return dataset
