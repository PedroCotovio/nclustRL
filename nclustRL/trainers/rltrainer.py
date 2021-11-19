from ray.rllib.agents import ppo, ddpg, sac

from nclustRL.utils.helper import loader

MODULES = [ppo, ddpg, sac]

# Algorithms PPO, Sac, DDPG, TD3


class NclustTrainer:

    def __init__(
            self,
            trainer,
            model,
            env,
            trainer_config,
            env_config,
            save_dir: str,
            checkpoints: bool = True,
            snapshots: bool = True
    ):

        self.trainer = loader(trainer, MODULES)
        self.model =

    def load(self):
        pass

    def load_from_torch(self):
        pass

    def save(self):
        pass

    def _has_checkpoint(self):
        pass

    def train(self):
        pass

    def tune(self):
        pass

    def plot(self):
        pass

    def test(self):
        pass



