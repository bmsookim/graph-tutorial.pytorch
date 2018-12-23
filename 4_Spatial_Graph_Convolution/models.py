import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttention

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, nouts, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] # concat
        for i, attention in enumerate(self.attentions):
            # Each attention will be GraphAttention(nfeat, nhid, dropout, concat=True)
            self.add_module('attention_1_{}'.format(i), attention)

        '''
        < Inductive Learning >
        self.attentions_2 = [GraphAttention(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)] # concat
        for i, attention in enumerate(self.attentions_2):
            # Each attention will be GraphAttention(nfeat, nhid, dropout, concat=True)
            self.add_module('attention_2_{}'.format(i), attention)
        '''
        self.out_att = [GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False) for _ in range(nouts)]
        for i, attention in enumerate(self.out_att):
            self.add_module('attention_out_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training) # in_dropout
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1) # concat
        #x = F.dropout(x, self.dropout, training=self.training)
        '''
        < Inductive learning >
        res = x
        x = F.dropout(x, self.dropout, training=self.training) # in_dropout
        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
        x = x + res
        '''
        x = torch.mean(torch.stack([att(x, adj) for att in self.out_att], dim=1), dim=1) # avg (for pubmed)

        return F.log_softmax(x, dim=1)
