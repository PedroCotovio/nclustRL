
from statistics import mean

import numpy as np

import nclustenv
from nclustenv.version import ENV_LIST

from nclustRL.utils.type_checker import is_trainer, is_env, is_dir, is_config
from nclustRL.utils.typing import RlLibTrainer, NclustEnvName, TrainerConfigDict, \
    Directory, Optional, SyntheticDataset


class Trainer:

    def __init__(
            self,
            trainer: RlLibTrainer,
            env: NclustEnvName,
            config: Optional[TrainerConfigDict] = None,
            save_dir: Optional[Directory] = None,
            seed: Optional[int] = None
    ):
        self._trainer = is_trainer(trainer)
        self._env = is_env(env)
        self._config = is_config(config)
        self._save_dir = is_dir(save_dir)
        self._seed = int(seed)
        self._np_random = np.random.RandomState(seed)

        self._agent = self.trainer(config=self.config, env=self.env)

    @property
    def trainer(self):
        return self._trainer

    @property
    def agent(self):
        return self._agent

    @property
    def env(self):
        return self._env

    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def seed(self):
        return self._seed

    def train(
            self,
            n_samples: Optional[int] = 1,
            checkpoint: Optional[str] = None,
            resume: Optional[bool] = False
    ):
        pass

    def load(self, checkpoint):

        self._agent = self.trainer(config=self.config, env=self.env).restore(checkpoint)

    def _compute_episode(self, env, obs):

        episode_reward = 0
        done = False

        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        episode_accuracy = 1.0 - env.volume_match

        return episode_reward, episode_accuracy

    def test(self, n_episodes: int = 100):

        env = nclustenv.make(self.env, **self.config['env_config'])

        accuracy = []
        reward = []

        for i in range(n_episodes):
            obs = env.reset()

            episode_reward, episode_accuracy = self._compute_episode(env, obs)

            accuracy.append(episode_accuracy)
            reward.append(episode_reward)

        return mean(reward), mean(accuracy)

    def _get_offline_env(self):

        if 'Offline' in self.env:
            return self.env
        else:
            for e in ENV_LIST:
                if e != self.env and self.env in e:
                    return e

    def test_dataset(self, dataset: SyntheticDataset):

        config = {
            'dataset': dataset,
            'train_test_split': 0.0,
            'seed': self.seed,
            'metric': self.config['env_config']['metric'],
            'action': self.config['env_config']['action'],
            'max_steps': self.config['env_config']['max_steps'],
            'error_margin': self.config['env_config']['error_margin'],
            'penalty': self.config['env_config']['penalty'],
        }

        env = nclustenv.make(self._get_offline_env(), **config)

        accuracy = []
        reward = []
        main_done = False

        while not main_done:
            obs, main_done = env.reset(train=False)

            episode_reward, episode_accuracy = self._compute_episode(env, obs)

            accuracy.append(episode_accuracy)
            reward.append(episode_reward)

        return reward, accuracy
