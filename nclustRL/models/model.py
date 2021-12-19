import dgl
import torch.nn as nn

from nclustRL.models import RGCN


class GraphEncoder(nn.Module):
    def __init__(self, n, conv_feats, n_classes, rel_names):
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

class RlModel:
    pass
