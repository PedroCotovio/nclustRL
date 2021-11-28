from nclustenv.version import ENV_LIST as _ENV_LIST
from nclustRL.utils import registry

from model import RlModel

# register envs in ray
for env in _ENV_LIST:
    registry.register_env(env)

registry.register_model('RlModel', RlModel)
