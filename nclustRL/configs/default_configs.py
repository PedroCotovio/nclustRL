from nclustRL.utils.typing import TrainerConfigDict
import torch
import os
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
import random


gpu_count = torch.cuda.device_count()
cpu_count = os.cpu_count()


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


PPO_PBT = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            # The GAE (lambda) parameter.
            "lambda": lambda: random.uniform(0.9, 1.0),
            # PPO clip parameter.
            "clip_param": lambda: random.uniform(0.01, 0.5),
            # Stepsize of SGD.
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            # Initial coefficient for KL divergence.
            "kl_coeff": lambda: random.uniform(0.2, 1),
            # Coefficient of the value function loss. IMPORTANT: you must tune this if
            # you set vf_share_layers=True inside your model's config.
            "vf_loss_coeff": lambda: random.uniform(0.5, 1.0),
            # Coefficient of the entropy regularizer.
            "entropy_coeff": lambda: random.uniform(0.0, 0.01),
            # Clip param for the value function. Note that this is sensitive to the
            # scale of the rewards. If your expected V is large, increase this.
            "vf_clip_param": lambda: random.uniform(10, 20),
            # Target value for KL divergence.
            "kl_target": lambda: random.uniform(0.003, 0.01),
            # Number of SGD iterations in each outer loop (i.e., number of epochs to
             # execute per train batch).
            "num_sgd_iter": lambda: random.randint(1, 30),
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            "train_batch_size": lambda: random.randint(1000, 160000),
        },
        custom_explore_fn=explore)

PPO_TUNE_INIT = {
    "num_sgd_iter": tune.choice([10, 20, 30]),
    "sgd_minibatch_size": tune.choice([128, 512, 2048]),
    "train_batch_size": tune.choice([10000, 20000, 40000])
}

MODEL_DEFAULTS = {
    # === Options for custom models ===
    # Name of a custom model to use
    "custom_model": 'general_model_torch',
    # Extra options to pass to the custom classes. These will be available to
    # the Model's constructor in the model_config field. Also, they will be
    # attempted to be passed as **kwargs to ModelV2 models. For an example,
    # see rllib/models/[tf|torch]/attention_net.py.
    "custom_model_config": {
        "fcnet_feats": [256, 256]
    },
    # Name of a custom action distribution to use.
    "custom_action_dist": None
}

TRAINER_DEFAULTS: TrainerConfigDict = {
    # === Settings for Rollout Worker processes ===
    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    "num_workers": (cpu_count - 1),
    # When `num_workers` > 0, the driver (local_worker; worker-idx=0) does not
    # need an environment. This is because it doesn't have to sample (done by
    # remote_workers; worker_indices > 0) nor evaluate (done by evaluation
    # workers; see below).
    "create_env_on_driver": False,
    # Number of environments to evaluate vector-wise per worker. This enables
    # model inference batching, which can improve performance for inference
    # bottlenecked workloads.
    "num_envs_per_worker": 1,
    # How to build per-Sampler (RolloutWorker) batches, which are then
    # usually concat'd to form the train batch. Note that "steps" below can
    # mean different things (either env- or agent-steps) and depends on the
    # `count_steps_by` (multiagent) setting below.
    # truncate_episodes: Each produced batch (when calling
    #   RolloutWorker.sample()) will contain exactly `rollout_fragment_length`
    #   steps. This mode guarantees evenly sized batches, but increases
    #   variance as the future return must now be estimated at truncation
    #   boundaries.
    # complete_episodes: Each unroll happens exactly over one episode, from
    #   beginning to end. Data collection will not stop unless the episode
    #   terminates or a configured horizon (hard or soft) is hit.
    "batch_mode": "truncate_episodes",

    # === Settings for the Trainer process ===
    # Discount factor of the MDP.
    "gamma": 0.99,
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE (lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    "rollout_fragment_length": int(1024 / (cpu_count - 1)),
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 1024,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,
    # Stepsize of SGD.
    "lr": 5e-5,
    # Learning rate schedule.
    "lr_schedule": None,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    "vf_loss_coeff": 1.0,

    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.0,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.01,

    # Arguments to pass to the policy optimizer. These vary by optimizer.
    "optimizer": {},

    # === Environment Settings ===
    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": None,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": False,
    # Don't set 'done' at the end of the episode.
    # In combination with `soft_horizon`, this works as follows:
    # - no_done_at_end=False soft_horizon=False:
    #   Reset env and add `done=True` at end of each episode.
    # - no_done_at_end=True soft_horizon=False:
    #   Reset env, but do NOT add `done=True` at end of the episode.
    # - no_done_at_end=False soft_horizon=True:
    #   Do NOT reset env at horizon, but add `done=True` at the horizon
    #   (pretending the episode has terminated).
    # - no_done_at_end=True soft_horizon=True:
    #   Do NOT reset env at horizon and do NOT add `done=True` at the horizon.
    "no_done_at_end": False,
    # The environment specifier:
    # This can either be a tune-registered env, via
    # `tune.register_env([name], lambda env_ctx: [env object])`,
    # or a string specifier of an RLlib supported type. In the latter case,
    # RLlib will try to interpret the specifier as either an openAI gym env,
    # a PyBullet env, a ViZDoomGym env, or a fully qualified classpath to an
    # Env class, e.g. "ray.rllib.examples.env.random_env.RandomEnv".
    "env": None,
    # The observation- and action spaces for the Policies of this Trainer.
    # Use None for automatically inferring these from the given env.
    "observation_space": None,
    "action_space": None,
    # Arguments dict passed to the env creator as an EnvContext object (which
    # is a dict plus the properties: num_workers, worker_index, vector_index,
    # and remote).
    "env_config": {},
    # If using num_envs_per_worker > 1, whether to create those new envs in
    # remote processes instead of in the same worker. This adds overheads, but
    # can make sense if your envs can take much time to step / reset
    # (e.g., for StarCraft). Use this cautiously; overheads are significant.
    "remote_worker_envs": False,
    # Timeout that remote workers are waiting when polling environments.
    # 0 (continue when at least one env is ready) is a reasonable default,
    # but optimal value could be obtained by measuring your environment
    # step / reset and model inference perf.
    "remote_env_batch_wait_ms": 0,
    # A callable taking the last train results, the base env and the env
    # context as args and returning a new task to set the env to.
    # The env must be a `TaskSettableEnv` sub-class for this to work.
    # See `examples/curriculum_learning.py` for an example.
    "env_task_fn": None,
    # If True, try to render the environment on the local worker or on worker
    # 1 (if num_workers > 0). For vectorized envs, this usually means that only
    # the first sub-environment will be rendered.
    # In order for this to work, your env will have to implement the
    # `render()` method which either:
    # a) handles window generation and rendering itself (returning True) or
    # b) returns a numpy uint8 image of shape [height x width x 3 (RGB)].
    "render_env": False,
    # If True, stores videos in this relative directory inside the default
    # output dir (~/ray_results/...). Alternatively, you can specify an
    # absolute path (str), in which the env recordings should be
    # stored instead.
    # Set to False for not recording anything.
    # Note: This setting replaces the deprecated `monitor` key.
    "record_env": False,
    # Whether to clip rewards during Policy's postprocessing.
    # None (default): Clip for Atari only (r=sign(r)).
    # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
    # False: Never clip.
    # [float value]: Clip at -value and + value.
    # Tuple[value1, value2]: Clip at value1 and value2.
    "clip_rewards": False,
    # If True, RLlib will learn entirely inside a normalized action space
    # (0.0 centered with small stddev; only affecting Box components).
    # We will unsquash actions (and clip, just in case) to the bounds of
    # the env's action space before sending actions back to the env.
    "normalize_actions": True,
    # Whether to use "rllib" or "deepmind" preprocessors by default
    # Set to None for using no preprocessor. In this case, the model will have
    # to handle possibly complex observations from the environment.
    "preprocessor_pref": None,

    # === Debug Settings ===
    # Set the ray.rllib.* log level for the agent process and its workers.
    # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # periodically print out summaries of relevant internal dataflow (this is
    # also printed out once at startup at the INFO level). When using the
    # `rllib train` command, you can also use the `-v` and `-vv` flags as
    # shorthand for INFO and DEBUG.
    "log_level": "DEBUG",
    # Whether to attempt to continue training if a worker crashes. The number
    # of currently healthy workers is reported as the "num_healthy_workers"
    # metric.
    "ignore_worker_failures": False,
    # Log system resource metrics to results. This requires `psutil` to be
    # installed for sys stats, and `gputil` for GPU metrics.
    "log_sys_usage": True,

    # === Deep Learning Framework Settings ===
    # tf: TensorFlow (static-graph)
    # tf2: TensorFlow 2.x (eager)
    # tfe: TensorFlow eager
    # torch: PyTorch
    "framework": "torch",
    # Enable tracing in eager mode. This greatly improves performance, but
    # makes it slightly harder to debug since Python code won't be evaluated
    # after the initial eager pass. Only possible if framework=tfe.
    "eager_tracing": False,

    # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    "explore": True,
    # Provide a dict specifying the Exploration object's config.
    "exploration_config": {
        "type": "StochasticSampling",
    },
    # === Evaluation Settings ===
    # Evaluate with every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    "evaluation_interval": None,
    # Number of episodes to run in total per evaluation period.
    # If using multiple evaluation workers (evaluation_num_workers > 1),
    # episodes will be split amongst these.
    # If "auto":
    # - evaluation_parallel_to_training=True: Will run as many episodes as the
    #   training step takes.
    # - evaluation_parallel_to_training=False: Error.
    "evaluation_num_episodes": 10,
    # Whether to run evaluation in parallel to a Trainer.train() call
    # using threading. Default=False.
    # E.g. evaluation_interval=2 -> For every other training iteration,
    # the Trainer.train() and Trainer.evaluate() calls run in parallel.
    # Note: This is experimental. Possible pitfalls could be race conditions
    # for weight synching at the beginning of the evaluation loop.
    "evaluation_parallel_to_training": False,
    # Internal flag that is set to True for evaluation workers.
    "in_evaluation": False,
    # Typical usage is to pass extra args to evaluation env creator
    # and to disable exploration by computing deterministic actions.
    # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
    # policy, even if this is a stochastic one. Setting "explore=False" here
    # will result in the evaluation workers not using this optimal policy!
    "evaluation_config": {
        "explore": False
    },
    # Number of parallel workers to use for evaluation. Note that this is set
    # to zero by default, which means evaluation will be run in the trainer
    # process (only if evaluation_interval is not None). If you increase this,
    # it will increase the Ray resource usage of the trainer since evaluation
    # workers are created separately from rollout workers (used to sample data
    # for training).
    "evaluation_num_workers": 0,
    # Customize the evaluation method. This must be a function of signature
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    # Trainer.evaluate() method to see the default implementation. The
    # trainer guarantees all eval workers have the latest policy state before
    # this function is called.
    "custom_eval_function": None,

    # === Advanced Rollout Settings ===
    # Use a background thread for sampling (slightly off-policy, usually not
    # advisable to turn on unless your env specifically requires it).
    "sample_async": False,

    # Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
    "observation_filter": "NoFilter",
    # Whether to synchronize the statistics of remote filters.
    "synchronize_filters": True,
    # Whether to LZ4 compress individual observations.
    "compress_observations": False,
    # Wait for metric batches for at most this many seconds. Those that
    # have not returned in time will be collected in the next train iteration.
    "collect_metrics_timeout": 180,
    # Smooth metrics over this many episodes.
    "metrics_smoothing_episodes": 100,
    # Minimum time per train iteration (frequency of metrics reporting).
    "min_iter_time_s": 0,
    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of train iterations.
    "timesteps_per_iteration": 0,
    # This argument, in conjunction with worker_index, sets the random seed of
    # each worker, so that identically configured trials will have identical
    # results. This makes experiments reproducible.
    "seed": None,
    # Any extra python env vars to set in the trainer process, e.g.,
    # {"OMP_NUM_THREADS": "16"}
    "extra_python_environs_for_driver": {},
    # The extra python environments need to set for worker processes.
    "extra_python_environs_for_worker": {},

    # === Resource Settings ===
    # Number of GPUs to allocate to the trainer process. Note that not all
    # algorithms can take advantage of trainer GPUs. Support for multi-GPU
    # is currently only available for tf-[PPO/IMPALA/DQN/PG].
    # This can be fractional (e.g., 0.3 GPUs).
    "num_gpus": 0.0001,
    # Set to True for debugging (multi-)?GPU funcitonality on a CPU machine.
    # GPU towers will be simulated by graphs located on CPUs in this case.
    # Use `num_gpus` to test for different numbers of fake GPUs.
    "_fake_gpus": False,
    # Number of CPUs to allocate per worker.
    "num_cpus_per_worker": 1,
    # Number of GPUs to allocate per worker. This can be fractional.
    "num_gpus_per_worker": (gpu_count - 0.0001) / (cpu_count - 1),
    # Any custom Ray resources to allocate per worker.
    "custom_resources_per_worker": {},
    # Number of CPUs to allocate for the trainer. Note: this only takes effect
    # when running in Tune. Otherwise, the trainer runs in the main program.
    "num_cpus_for_driver": 1,
    # The strategy for the placement group factory returned by
    # `Trainer.default_resource_request()`. A PlacementGroup defines, which
    # devices (resources) should always be co-located on the same node.
    # For example, a Trainer with 2 rollout workers, running with
    # num_gpus=1 will request a placement group with the bundles:
    # [{"gpu": 1, "cpu": 1}, {"cpu": 1}, {"cpu": 1}], where the first bundle is
    # for the driver and the other 2 bundles are for the two workers.
    # These bundles can now be "placed" on the same or different
    # nodes depending on the value of `placement_strategy`:
    # "PACK": Packs bundles into as few nodes as possible.
    # "SPREAD": Places bundles across distinct nodes as even as possible.
    # "STRICT_PACK": Packs bundles into one node. The group is not allowed
    #   to span multiple nodes.
    # "STRICT_SPREAD": Packs bundles across distinct nodes.
    "placement_strategy": "PACK",

    # === Logger ===
    # Define logger-specific configuration to be used inside Logger
    # Default value None allows overwriting with nested dicts
    "logger_config": None,
    # Experimental flag.
    # If True, no (observation) preprocessor will be created and
    # observations will arrive in model as they are returned by the env.
    # In the future, the default for this will be True.
    "_disable_preprocessor_api": True,
}

DEFAULT_CONFIG = TRAINER_DEFAULTS.copy()
DEFAULT_CONFIG['model'] = MODEL_DEFAULTS