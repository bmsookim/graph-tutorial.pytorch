# ************************************************************
# Author : Bumsoo Kim, 2018
# Github : https://github.com/meliketoy/graph-tutorial.pytorch
#
# Korea University, Data-Mining Lab
# Graph Convolutional Neural Network
#
# Description : train.py
# The main code for training Graph Attention Networks.
# ***********************************************************

import time
import random
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from utils import *
from models import GAT
from opts import TrainOptions

"""
N : number of nodes
D : number of features per node
E : number of classes

@ input :
    - adjacency matrix (N x N)
    - feature matrix (N x D)
    - label matrix (N x E)

@ dataset :
    - citeseer
    - cora
    - pubmed
"""
opt = TrainOptions().parse()

# Data upload
adj, features, labels, idx_train, idx_val, idx_test = load_data(path=opt.dataroot, dataset=opt.dataset)
use_gpu = torch.cuda.is_available()

model, optimizer = None, None
best_acc = 0
early_stop = 0

# Define the model and optimizer
if (opt.model == 'attention'):
    print("| Constructing Graph Attention Network model...")
    model = GAT(
            nfeat = features.shape[1],
            nhid = opt.num_hidden,
            nclass = int(labels.max().item()) + 1,
            dropout = opt.dropout,
            nheads = opt.nb_heads,
            nouts = opt.nb_outs,
            alpha = opt.alpha
    )
else:
    raise NotImplementedError

if (opt.optimizer == 'sgd'):
    optimizer = optim.SGD(
            model.parameters(),
            lr = opt.lr,
            weight_decay = opt.weight_decay,
            momentum = 0.9
    )
elif (opt.optimizer == 'adam'):
    optimizer = optim.Adam(
            model.parameters(),
            lr = opt.lr,
            weight_decay = opt.weight_decay
    )
else:
    raise NotImplementedError

if use_gpu:
    model.cuda()
    features, adj, labels, idx_train, idx_val, idx_test = \
        list(map(lambda x: x.cuda(), [features, adj, labels, idx_train, idx_val, idx_test]))

features, adj, labels = list(map(lambda x : Variable(x), [features, adj, labels]))

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

save_point = os.path.join('./checkpoint', opt.dataset)

if not os.path.isdir(save_point):
    os.mkdir(save_point)

# Train
def train(epoch):
    global best_acc, early_stop

    t = time.time()
    model.train()
    optimizer.lr = opt.lr
    optimizer.zero_grad()

    output = model(features, adj)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    # Validation for each epoch
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if acc_val > best_acc:
        best_acc = acc_val
        state = {
            'model': model,
            'acc': best_acc,
            'epoch': epoch,
        }

        torch.save(state, os.path.join(save_point, '%s.t7' %(opt.model)))
        early_stop = 0
    else:
        early_stop += 1

        if (early_stop > 100):
            return True

    sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.write("=> Training Epoch #{} : lr = {:.4f}".format(epoch, optimizer.lr))
    sys.stdout.write(" | Training acc : {:6.2f}%".format(acc_train.data.cpu().numpy() * 100))
    sys.stdout.write(" | Best acc : {:.2f}%". format(best_acc.data.cpu().numpy() * 100))
    return False

# Main code for training
if __name__ == "__main__":
    print("\n[STEP 2] : Obtain (adjacency, feature, label) matrix")
    print("| Adjacency matrix : {}".format(adj.shape))
    print("| Feature matrix   : {}".format(features.shape))
    print("| Label matrix     : {}".format(labels.shape))

    # Training
    print("\n[STEP 3] : Training")
    for epoch in range(1, opt.epoch+1):
        stopped = train(epoch)
        if (stopped):
            break

    print("\n=> Training finished!")
