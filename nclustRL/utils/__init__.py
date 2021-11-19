
from nclustenv.version import ENV_LIST
import registry

# register envs in ray
for env in ENV_LIST:
    registry.register_env(env)

# register models

# registry.register_model('RlModel')
