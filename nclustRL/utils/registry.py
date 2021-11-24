from ray.tune.registry import register_env as register
from ray.rllib.models import ModelCatalog
import nclustenv


def register_env(id):
    return register(id, lambda config: nclustenv.make(id, **config))


def register_model(id, model):
    return ModelCatalog.register_custom_model(id, model)
