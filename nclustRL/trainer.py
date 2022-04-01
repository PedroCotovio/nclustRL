
from os import path
from statistics import mean, stdev
import numpy as np

import nclustenv
from nclustenv.version import ENV_LIST
import ray
from ray.tune.stopper import Stopper
from tqdm import tqdm
import nclustRL

from nclustRL.utils.type_checker import is_trainer, is_env, is_file, is_config, is_dataset
from nclustRL.utils.typing import RlLibTrainer, NclustEnvName, TrainerConfigDict, \
    Directory, Optional, SyntheticDataset

from gym.wrappers import TransformObservation
from nclustRL.utils.helper import transform_obs, randint

from nclustRL.configs.default_configs import DEFAULT_CONFIG

from nclustenv.configs import biclustering, triclustering

from time import perf_counter as counter


class Trainer:

    def __init__(
            self,
            trainer: RlLibTrainer,
            env: NclustEnvName,
            name: Optional[str] = 'test',
            config: Optional[TrainerConfigDict] = None,
            save_dir: Optional[Directory] = '',
            seed: Optional[int] = 7
    ):

        if config is None:
            config = DEFAULT_CONFIG

        self._trainer = is_trainer(trainer)
        self._env = is_env(env)
        self._name = str(name)
        self._config = is_config(config)
        self._dir = str(save_dir)
        self._seed = int(seed)
        self._np_random = np.random.RandomState(seed)

        self._config['env'] = self.env

        if not self._config['env_config']:
            self._config['env_config'] = self._get_default_env_config()

        self._agent = self.trainer(config=self.eval_config, env=self.env)

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

    @property
    def eval_config(self):
        eval_dict = self.config['evaluation_config']
        config = self.config.copy()
        config.update(eval_dict)

        return config

    @property
    def dim(self):
        if 'Bicluster' in self.env:
            return 2
        return 3

    @staticmethod
    def restart_ray():
        nclustRL.restart_cluster()

    def _get_default_env_config(self):
        if self.dim == 2:
            config = biclustering
        else:
            config = triclustering

        return config.binary.base

    def _set_seed(self, seed):

        config = self.config.copy()
        config['env_config']['seed'] = seed
        config['seed'] = seed

        return config

    def train(
            self,
            n_samples: Optional[int] = 1,
            metric: Optional[str] = 'episode_reward_mean',
            mode: Optional[str] = 'max',
            checkpoint_freq: Optional[int] = 10,
            stop: Optional[Stopper] = None,
            stop_iters: Optional[int] = 100,
            stop_metric: Optional[float] = 10,
            checkpoint: Optional[str] = None,
            resume: Optional[bool] = False,
            verbose: Optional[int] = 1,
            *args, **kwargs
    ):
        if checkpoint:
            checkpoint = is_file(checkpoint)

        generator = ((i, self._np_random.randint(0, 1000)) for i in range(n_samples))

        results = []

        with tqdm(generator, unit='sample') as tsample:

            for i, seed in tsample:

                tsample.set_description(f"Sample {i + 1}")

                local_dir = path.join(self.save_dir, 'sample_{}'.format(i))

                if not stop:
                    stop_criteria = {
                        "training_iteration": stop_iters,
                        metric: stop_metric,
                    }

                else:
                    stop_criteria = stop

                # Update seeds
                config = self._set_seed(seed)

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
                    verbose=verbose,
                    *args, **kwargs
                )

                checkpoint = analysis.get_best_checkpoint(
                    trial=analysis.get_best_trial(
                        metric=metric,
                        mode=mode), metric=metric, mode=mode)

                results.append({
                    'config': analysis.get_best_config(metric=metric, mode=mode),
                    'path': checkpoint,
                    'metric': analysis.best_result,
                    'df': analysis.dataframe(metric=metric, mode=mode)
                })

                best_checkpoint = results[np.argmax([res['metric'] for res in results])]

                tsample.set_postfix(metric=best_checkpoint['metric'])

        return best_checkpoint

    def load(self, checkpoint):

        checkpoint = is_file(checkpoint)
        self._agent.restore(checkpoint)

    def _compute_episode(self, env, obs):

        episode_reward = 0
        done = False

        while not done:
            action = self.agent.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        episode_accuracy = 1.0 - env.volume_match

        return episode_reward, episode_accuracy

    @staticmethod
    def _wrap_env(env):
        return TransformObservation(env, transform_obs)

    def test(self, n_episodes: int = 100, verbose=True):

        start_time = counter()

        n_episodes = int(n_episodes)

        env = self._wrap_env(nclustenv.make(self.env, **self.config['env_config']))

        accuracy = []
        reward = []
        time = []

        for i in range(n_episodes):

            start_time = counter()

            obs = env.reset()

            episode_reward, episode_accuracy = self._compute_episode(env, obs)

            accuracy.append(episode_accuracy)
            reward.append(episode_reward)

            end_time = counter()
            time.append(end_time - start_time)

            if verbose:
                print('Episode {} of {} done.'.format(i+1, n_episodes))

        return np.array([[mean(reward), stdev(reward)], [mean(accuracy), stdev(accuracy)], [mean(time), stdev(time)]])

    def make_env(self):

        return self._wrap_env(nclustenv.make(self.env, **self.config['env_config']))


    def _get_offline_env(self):

        if 'Offline' in self.env:
            return self.env
        else:
            for e in ENV_LIST:
                if e != self.env and self.env in e:
                    return e

    def test_dataset(self, dataset: SyntheticDataset, verbose=True):

        config = {
            'dataset': is_dataset(dataset),
            'train_test_split': 0.0,
            'seed': self.seed,
            'metric': self.config['env_config'].get('metric'),
            'action': self.config['env_config'].get('action'),
            'max_steps': self.config['env_config'].get('max_steps'),
            'error_margin': self.config['env_config'].get('error_margin'),
            'penalty': self.config['env_config'].get('penalty'),
        }

        env = self._wrap_env(nclustenv.make(self._get_offline_env(), **config))

        accuracy = []
        reward = []
        main_done = False
        i = 1

        while not main_done:
            obs, main_done = env.reset(train=False)

            episode_reward, episode_accuracy = self._compute_episode(env, obs)

            accuracy.append(episode_accuracy)
            reward.append(episode_reward)

            if verbose:
                print('Episode {} done.'.format(i))
                i += 1

        return reward, accuracy
