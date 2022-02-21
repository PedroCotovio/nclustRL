from collections.abc import Iterable
from ray.tune import grid_search
import numpy as np
from nclustRL.utils.typing import Dict
import torch as th
import itertools

import dgl
import nclustenv
from nclustenv.configs import biclustering, triclustering
from gym.wrappers import TransformObservation
import ray
from time import sleep


def loader(cls, module=None):

    """Loads a method from a pointer or a string"""

    if module is None:
        module = []

    if not isinstance(module, Iterable):
        var = module
        module = [var]

    for m in module:
        try:
            return getattr(m, cls) if isinstance(cls, str) else cls

        except AttributeError:
            pass

    raise AttributeError('modules {} have no attribute {}'.format(module, cls))


def random_rollout(env):

    state = env.reset()

    done = False
    cumulative_reward = 0

    # Keep looping as long as the simulation has not finished.
    while not done:
        # Choose a random action (either 0 or 1).
        action = env.action_space.sample()

        # Take the action in the environment.
        state, reward, done, _ = env.step(action)

        # Update the cumulative reward.
        cumulative_reward += reward

    # Return the cumulative reward.
    return cumulative_reward


def inherit_dict(parent: Dict, child: Dict):

    res = parent.copy()
    res.update(child)

    return res


def grid_interval(min, max, interval=5):

    dtype = None

    if isinstance(max, int) and isinstance(min, int):
        dtype = 'int64'

    return grid_search(np.linspace(min, max, interval, dtype=dtype))


def transform_obs(obs):

    obs = obs.copy()
    state = obs['state'].clone()
    ntypes = state.ntypes
    keys = sorted(list(state.nodes[ntypes[0]].data.keys()))

    ndata = {}

    for ntype in ntypes:
        ndata[ntype] = th.vstack(
            [th.where(
                state.ndata[key][ntype], 
                th.ones(state.ndata[key][ntype].shape, dtype=th.float32), 
                th.full(state.ndata[key][ntype].shape, -1, dtype=th.float32)
                ) for key in keys]
        ).transpose(0, 1)

        state.nodes[ntype].data.clear()

    state.ndata['feat'] = ndata
    obs['state'] = state

    return obs

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b) 

def randint(size, dtype):
    return th.randint(low=0, high=2, size=[size], dtype=dtype)


def generate_dummy_obs(batch_size, dim):

    if dim == 2:
        env_name = 'BiclusterEnv-v0'
        config_module = biclustering
    else:
        env_name = 'TriclusterEnv-v0'
        config_module = triclustering

    env = TransformObservation(nclustenv.make(env_name, **config_module.binary.base), transform_obs)

    return dgl.batch([env.reset()['state'] for _ in range(batch_size)])


def restart_cluster():
    ray.shutdown()
    sleep(0.1)
    ray.init()





