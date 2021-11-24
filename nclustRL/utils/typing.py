from typing import Callable
from typing import Optional
from ray.rllib.utils.typing import TrainerConfigDict
from nclustenv.utils.datasets import SyntheticDataset
from ray.rllib.agents.trainer import Trainer

RlLibTrainer = Callable
NclustEnvName = str
Directory = str

