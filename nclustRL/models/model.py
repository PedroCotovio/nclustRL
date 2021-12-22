import dgl
import torch.nn as nn
import torch as th

from nclustRL.models import RGCN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX
from nclustRL.utils.helper import pairwise


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


class FcNet(nn.Module):

    def __init__(self, n, feats):

        _layers = []
        feats.insert(0, n)

        for in_feats, out_feats in pairwise(feats):
            _layers.append(SlimFC(
                        in_size=in_feats,
                        out_size=out_feats,
                        initializer=normc_initializer(1.0),
                        activation_fn='tanh'))

        self._hidden_layers = nn.Sequential(*_layers)

    def forward(self, hg):
        self._hidden_layers(hg)


class RlModel(TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self._n = obs_space['state'].n
        etypes = ['elem']
        etypes = etypes * 3 if obs_space['state'].shape[0] > 2 else etypes

        conv_feats = list(model_config["custom_model_config"].get("conv_feats", [64 if self._n < 64 else 128]))
        fcnet_feats = list(model_config["custom_model_config"].get("fcnet_feats", [256, 256]))

        gencoder = GraphEncoder(self._n, conv_feats, etypes)
        fcnet = FcNet(conv_feats[-1], fcnet_feats)

        self._action_branch = SlimFC(
                        in_size=fcnet_feats[-1],
                        out_size=self._n,
                        initializer=normc_initializer(0.01),
                        activation_fn=None)

        self._param_branch = SlimFC(
            in_size=fcnet_feats[-1],
            out_size=(num_outputs - self._n),
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self._hidden_layers = nn.Sequential(gencoder, fcnet)

        # Value Branch
        self._value_branch = SlimFC(
            in_size=fcnet_feats[-1],
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["state"]
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        self._features = self._hidden_layers(obs)

        _actions = self._action_branch(self._features)
        _params = self._param_branch(self._features)

        intent_vector = th.unsqueeze(_actions, 1)
        action_logits = th.sum(avail_actions * intent_vector, dim=2)
        inf_mask = th.clamp(th.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return th.cat((action_logits + inf_mask), _params)

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        self._value_branch(self._features).squeeze(1)
