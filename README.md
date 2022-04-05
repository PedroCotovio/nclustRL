# NclustRL

NclustRL is a toolkit that implements some functionalities to help train agents for n-clustering tasks. It works with 
Ray's [RLlib](https://github.com/ray-project/ray/tree/master/rllib) to train DRL agents.

Ray is a general-purpose framework for distributed computing that implements a known library for hyperparameter tunning, 
Tune. Furthermore, it implements RLlib, a DRL framework that supports distributed computing and great customization.

NclustRL implements a trainer API for n-clustering that handles all training tasks for the user; a set 
of default models and metrics; and other helpful functions. Likewise, it provides a set of default configurations for 
$n$-clustering tasks available in "configs".

![Diagram exemplifying NclustEnv's architecture](diagNclustRL.png)

The trainer API aims to provide a simple way of training and testing DRL agents for n-clustering tasks. This class handles 
all of RLlib's logic and expose only user-friendly methods.

After initialized, the trainer exposes four primary methods:

* **Train:** Exposes the primary training function. It receives the training parameters that should be passed on to 
*Tune*, initiates the training process, manages multiple samples of the same trial, and parses results returning the 
best performance obtained;
* **Load:** Imports an agent from a checkpoint for testing;
* **Test:** Evaluates the accuracy and mean reward and returns the mean and standard deviation for each of these metrics 
across n episodes.
* **Test Dataset:** Evaluates the performance in the same way as *Test* but receives as input a specific dataset from 
where episodes should be sampled.

## Installation

This tool can be installed from PyPI:

```sh
pip install nclustRL
```

## Getting started

Here are the basics, for more information check the Experiments available on "Exp".

```python

## Train basic agent

from nclustRL.trainer import Trainer
from nclustRL.configs.default_configs import PPO_PBT, DEFAULT_CONFIG
from ray.rllib.agents.ppo import PPOTrainer
from nclustenv.configs import biclustering

# Inicialize Trainer

config = DEFAULT_CONFIG.copy()
config['env_config'] = biclustering.binary.basic_v2

trainer = Trainer(
    trainer=PPOTrainer,
    env='BiclusterEnv-v0',
    save_dir='nclustRL/Exp/test',
    name='test',
    config=config
)

## Tune agent

best_checkpoint = trainer.train(
    num_samples=8, 
    scheduler=PPO_PBT,
    stop_iters=500,
)

```

### Model

By default this tool implements a model for hybrid proximal policy optimization algorithm, available in "models". This 
model can be customized, or other models might be implemented and passed in the configs.

## License
[GPLv3](LICENSE)
