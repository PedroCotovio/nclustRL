import dgl
import torch.nn as nn

from nclustRL.models import RGCN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class GraphEncoder(nn.Module):
    def __init__(self, n, conv_feats, rel_names):
        super().__init__()

        conv_feats.insert(0, n)
        self.rgcn = RGCN(conv_feats, rel_names)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h, 'w')
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0
            for ntype in h.keys():
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            
            return hg

class RlModel(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        n = obs_space['state'].n

        # TODO select default network vals


        conv_feats = list(model_config["custom_model_config"].get("conv_feats", [50]))
        fcnet_feats = list(model_config["custom_model_config"].get("fcnet_feats", []))
        


    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)
