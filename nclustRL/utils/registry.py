from ray.tune.registry import register_env as register
from ray.rllib.models import ModelCatalog
from gym.wrappers import TransformObservation
from nclustRL.utils.helper import transform_obs
import nclustenv


def register_env(id):
    return register(id, lambda config: TransformObservation(nclustenv.make(id, **config), transform_obs))


def register_model(id, model):
    return ModelCatalog.register_custom_model(id, model)
