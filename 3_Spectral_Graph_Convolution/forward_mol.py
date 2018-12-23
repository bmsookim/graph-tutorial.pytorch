import math
import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from molecule_utils import atom_features, bond_features, num_atom_features, num_bond_features

# Graph Convolution Layer
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, edge_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor))
        else:
            self.register_parameter('bias', None)

        if init=='uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init=='xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init=='kaiminig':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    # Initializations
    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    # Forward
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# Graph Convolution Network Model
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nedges, dropout, init):
        super(GCN, self).__init__()

        self.weight = Parameter(nn.init.xavier_normal_(torch.Tensor(nfeat, nhid).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.gc1 = [GraphConvolution(nhid, nhid, nedges, init=init) for _ in range(nedges)]
        for idx, gc1 in enumerate(self.gc1):
            self.add_module('GCN_1_%d' %idx, gc1)
        self.gc2 = [GraphConvolution(nhid, nclass, nedges, init=init) for _ in range(nedges)]
        for idx, gc2 in enumerate(self.gc2):
            self.add_module('GCN_2_%d' %idx, gc2)

        self.dropout = dropout
        self.adj_dropout = dropout

    def AGG(self, x1, x2, method='add'):
        res = None

        if method == 'add':
            res = x1 + x2
        elif method == 'max':
            res = max(x1, x2)
        elif method == 'min':
            res = min(x1, x2)
        else:
            raise NotImplementedError

        return res

    def forward(self, x, adj):
        # iterate through the edges
        prev = None
        in_x = torch.matmul(x,self.weight)

        for i in range(adj.shape[2]):
            x = F.leaky_relu(self.gc1[i](F.dropout(in_x,self.dropout,self.training), F.dropout(adj[:,:,i], self.adj_dropout, self.training)), negative_slope = 0.2, inplace = False)
            x = F.leaky_relu(self.gc2[i](F.dropout(x,self.dropout,self.training), F.dropout(adj[:,:,i], self.adj_dropout, self.training)), 0.2)

            if i == 0:
                prev = x
            else:
                prev = self.AGG(prev, x)

        return prev

if __name__ == "__main__":
    print("Forward path for molecule GCN processing")

    # name = 1-benzylimidazole
    # pid = BRD-K32795028
    smiles = 'c1ccc(Cn2ccnc2)cc1'
    mol = Chem.MolFromSmiles(smiles)

    num_atom = max([atom.GetIdx() for atom in mol.GetAtoms()]) + 1
    adj = np.zeros((num_atom, num_atom, num_bond_features()))

    features = [atom_features(atom) * 1 for atom in mol.GetAtoms()]
    feature = np.stack(features, axis=0)

    edge_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_features(bond)) for bond in mol.GetBonds()]

    for edge in edge_list:
        v1, v2, f = edge
        f = f * 1
        adj[v1][v2] = f
        adj[v2][v1] = f

    adj, feature = list(map(lambda x : torch.FloatTensor(x), [adj, feature]))
    if torch.cuda.is_available():
        adj, feature = list(map(lambda x : x.cuda(), [adj, feature]))

    print("Input Feature (# nodes x 62): " + str(feature.shape)) # node x 62
    print("Input adjacency (# nodes x # nodes x 6): " + str(adj.shape)) # node x node x 6

    model = GCN(
            nfeat = num_atom_features(),
            nhid = 100,
            nclass = 500,
            nedges = num_bond_features(),
            dropout = 0.5,
            init = 'xavier'
    )

    output = model(feature, adj)
    print("Output (# of nodes x 500): " + str(output.shape))

    max_lst, idx = ((torch.max(output, dim=0)))
    print("Max pool to a 500 dimensional vector : "+ str(max_lst.shape))
