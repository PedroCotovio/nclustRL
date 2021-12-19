import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from nclustRL.utils.helper import pairwise

class HeteroRelu(nn.ReLU):

    def __init__(self, inplace:bool = False):
        super(HeteroRelu, self).__init__(inplace=inplace)

    def forward(self, inputs):
        
        return {k: super(HeteroRelu, self).forward(v) for k, v in inputs.items()}

class GraphSequential(nn.Sequential):

    def __init__(self, *args):
        super(GraphSequential, self).__init__(*args)

    def forward(self, graph, feat, edge_weight=None):
        for module in self:

            if isinstance(module, dglnn.HeteroGraphConv):

                rel_names = zip(module.mods.keys(), graph.canonical_etypes)
                feat = module(
                    g=graph, 
                    inputs=feat, 
                    mod_kwargs={
                        rel: dict(edge_weight=graph.edges[canonical].data[edge_weight]) 
                        for rel, canonical in rel_names})

            else:
                feat = module(inputs=feat)

        return feat


class RGCN(nn.Module):
    def __init__(self, layers, rel_names):
        super().__init__()

        _layers = []

        for in_feats, out_feats in pairwise(layers): 

            _layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, out_feats)
                for rel in rel_names}, aggregate='sum'))

            _layers.append(HeteroRelu())

        self._hidden_layers = GraphSequential(*_layers)

            

    def forward(self, graph, feat, edge_weight=None):

        return self._hidden_layers(graph, feat, edge_weight)


class HeteroClassifier(nn.Module):
    def __init__(self, n, conv_feats, n_classes, rel_names):
        super().__init__()

        conv_feats.insert(0, n)

        self.rgcn = RGCN(conv_feats, rel_names)
        self.classify = nn.Linear(conv_feats[-1], n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h, 'w')
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0
            for ntype in h.keys():
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)

            return self.classify(hg)