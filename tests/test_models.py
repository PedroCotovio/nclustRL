from nclustRL.models.model import HeteroClassifier
import torch
from torch.nn import functional as F
from tqdm import tqdm
from nclustenv.datasets.biclustering.binary import base
from dgl.dataloading import GraphDataLoader

import torch
from torch.nn import functional as F
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
import torch as th
import dgl

    def loader(cls, module=None):

        return getattr(module, cls) if isinstance(cls, str) else cls
    
    def dense_to_dgl(x, device, cuda=0, nclusters=1, clust_init='zeros', duplicate=True):

        # set (u,v)
        clust_init = loader(th, clust_init)

        tensor = th.tensor([[i, j, elem] for i, row in enumerate(x) for j, elem in enumerate(row)]).T

        if duplicate:

            graph_data = {
                ('row', 'elem', 'col'): (tensor[0].int(), tensor[1].int()),
                ('col', 'elem', 'row'): (tensor[1].int().detach().clone(), tensor[2].int().detach().clone()),
                }

            # create graph
            G = dgl.heterograph(graph_data)

            # set weights
            G.edges[('row', 'elem', 'col')].data['w'] = tensor[2].float()
            G.edges[('col', 'elem', 'row')].data['w'] = tensor[2].float()

        else:

            graph_data = {
                ('row', 'elem', 'col'): (tensor[0].int(), tensor[1].int()),
                }

            # create graph
            G = dgl.heterograph(graph_data)

            # set weights
            G.edges[('row', 'elem', 'col')].data['w'] = tensor[2].float()

        # set cluster members

        for n, axis in enumerate(['row', 'col']):
            for i in range(nclusters):
                G.nodes[axis].data[i] = th.randint(0, 2, (x.shape[n],), dtype=torch.bool)

        ndata = {}
        ntypes = G.ntypes
        keys = sorted(list(G.nodes[ntypes[0]].data.keys()))

        for ntype in ntypes:
            ndata[ntype] = torch.vstack(
                [G.ndata[key][ntype].float() for key in keys]
            ).transpose(0, 1)

            G.nodes[ntype].data.clear()

        G.ndata['feat'] = ndata

        if device == 'gpu':
            G = G.to('cuda:{}'.format(cuda))

        return G

import nclustenv
import torch



def test_embedings(graphs):

    batch_size=1
    shuffle=True
    nclasses = 5
    n = 5

    # dataloader = GraphDataLoader(
    #     base,
    #     batch_size=batch_size,
    #     drop_last=False,
    #     shuffle=shuffle)

    etypes = graphs[0].etypes

    model = HeteroClassifier(n, [n*2], nclasses, etypes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())


    for epoch in range(20):
        with tqdm(graphs, unit="batch") as tepoch:
            for batched_graph in tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                # batched_graph = transform_obs(n, batched_graph)
                labels = torch.randint(0, 4, (batch_size,)).to('cuda:0')

                logits = model(batched_graph)
                loss = F.cross_entropy(logits, labels)

                predictions = logits.argmax(dim=1, keepdim=True).squeeze()
                correct = (logits == labels).sum().item()

                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = correct / batch_size
                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)



if __name__ == '__main__ ':

    env = nclustenv.make('BiclusterEnv-v0', **dict(shape=[[100, 10], [110, 15]], clusters=[5,5]))

graphs_dup = []
graphs = []
for i in range(10):
    env.reset()
    X = env.state._generator.X
    graphs_dup.append(dense_to_dgl(X, device='gpu', nclusters=5))
    graphs.append(dense_to_dgl(X, device='gpu', nclusters=5, duplicate=False))

    test_embedings()