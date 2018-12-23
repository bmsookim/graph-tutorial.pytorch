# ************************************************************
# Author : Bumsoo Kim, 2018
# Github : https://github.com/meliketoy/graph-tutorial.pytorch
#
# Korea University, Data-Mining Lab
# Basic Tutorial for Non-Euclidean Graph Representation Learning
#
# Description : utils.py
# Code for uploading planetoid dataset
# ***********************************************************

import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from scipy.sparse import csgraph

def parse_index_file(filename):
    index = []

    for line in open(filename):
        index.append(int(line.strip()))

    return index

def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end+1)).difference(L))

def normalize_sparse_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def normalize_sparse_adj(mx):
    """Laplacian Normalization"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(path="/home/bumsoo/Data/Planetoid", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)

    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)

    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if (sys.version_info > (3,0)):
                objects.append(pkl.load(f, encoding='latin1')) # python3 compatibility
            else:
                objects.append(pkl.load(f)) # python2

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx)

    if dataset == 'citeseer':
        #Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx), max(test_idx)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    # Feature & Adjacency Matrix
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(int(adj.sum().sum()/2 + adj.diagonal().sum()/2)))

    # Normalization
    features = normalize_sparse_features(features)
    adj = normalize_sparse_adj(adj + sp.eye(adj.shape[0])) # Input is A_hat
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test
