from collections.abc import Iterable


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


