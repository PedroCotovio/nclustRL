
from os import path
from statistics import mean
import numpy as np

import nclustenv
from nclustenv.version import ENV_LIST
from ray.util.client import ray

from nclustRL.utils.type_checker import is_trainer, is_env, is_dir, is_config, is_dataset
from nclustRL.utils.typing import RlLibTrainer, NclustEnvName, TrainerConfigDict, \
    Directory, Optional, SyntheticDataset


class Trainer:

    def __init__(
            self,
            trainer: RlLibTrainer,
            env: NclustEnvName,
            name: Optional[str] = 'test',
            config: Optional[TrainerConfigDict] = None,
            save_dir: Optional[Directory] = None,
            seed: Optional[int] = None
    ):
        self._trainer = is_trainer(trainer)
        self._env = is_env(env)
        self._name = str(name)
        self._config = is_config(config)
        self._dir = str(save_dir)
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
        return path.join(self._dir, self._name)

    @property
    def seed(self):
        return self._seed

    def train(
            self,
            n_samples: Optional[int] = 1,
            metric: Optional[str] = 'episode_reward_mean',
            mode: Optional[str] = None,
            checkpoint_freq: Optional[int] = 10,
            stop_iters: Optional[int] = 1000,
            stop_metric: Optional[float] = None,
            checkpoint: Optional[str] = None,
            resume: Optional[bool] = False,
            verbose: Optional[int] = 1
    ):
        if checkpoint:
            checkpoint = is_dir(checkpoint)

        generator = ((i, self._np_random.randint(0, 1000)) for i in range(n_samples))

        results = []

        for i, seed in generator:

            local_dir = path.join(self.save_dir, 'sample_{}'.format(i))

            stop_criteria = {
                "training_iteration": stop_iters,
                metric: stop_metric,
            }

            # Update seed
            config = self.config
            config['env_config']['seed'] = seed

            analysis = ray.tune.run(
                self.trainer,
                config=config,
                local_dir=local_dir,
                metric=metric,
                mode=mode,
                stop=stop_criteria,
                checkpoint_at_end=True,
                checkpoint_freq=checkpoint_freq,
                resume=resume,
                restore=checkpoint,
                queue_trials=True,
                verbose=verbose
            )

            checkpoints = analysis.get_trial_checkpoints_paths(
                trial=analysis.get_best_trial(
                    metric=metric,
                    mode=mode), metric=metric)

            results.append({
                'config': analysis.get_best_config(metric=metric, mode=mode),
                'path': checkpoints[0][0],
                'metric': checkpoints[0][1],
            })

        best_checkpoint = results[np.argmax([res['metric'] for res in results])]

        return best_checkpoint

    def load(self, checkpoint):

        checkpoint = is_dir(checkpoint)

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

        n_episodes = int(n_episodes)

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
            'dataset': is_dataset(dataset),
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
