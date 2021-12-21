from nclustRL.utils import registry

from .pre_train import RGCN, HeteroClassifier
from .model import GraphEncoder, RlModel


registry.register_model('main_model_torch', RlModel)
