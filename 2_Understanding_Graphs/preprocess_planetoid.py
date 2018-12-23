# ************************************************************
# Author : Bumsoo Kim, 2018
# Github : https://github.com/meliketoy/graph-tutorial.pytorch
#
# Korea University, Data-Mining Lab
# Basic Tutorial for Non-Euclidean Graph Representation Learning
#
# Description : preprocess.py
# Code for preprocessing planetoid dataset
# ***********************************************************

import torch
import pickle as pkl
import argparse
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from pathlib import Path
from load_planetoid import read_data

def parse_index_file(filename):
    index = []

    for line in open(filename):
        index.append(int(line.strip()))

    return index

def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end+1)).difference(L))

def preprocess_citeseer(tx, ty, test_idx):
    #Citeseer dataset contains some isolated nodes in the graph
    bef_tx = (tx.todense())
    test_idx_range = np.asarray(test_idx, dtype=np.int64)
    test_idx_range_full = range(min(test_idx), max(test_idx)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended

    if (args.step == 'isolate'):
        print("Citeseer 전처리 이전 test shape : %s" %str(bef_tx.shape))
        print("Citeseer 전처리 이후 test shape : %s" %str(tx.todense().shape))

    ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

    return tx, ty

def normalize_sparse_feature(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # D_hat (Diagonal Matrix for Degrees)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten() # D_hat^(-1/2)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt) # list of diagonal of D_hat^(-1/2)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo() # D_hat^(-1/2) . A_hat . D_hat^(-1/2)

def check_symmetric(a, tol=1e-8):
    return not False in (np.abs(a-a.T) < tol)

def pitfall(path, dataset):
    x, y, tx, ty, allx, ally, graph = read_data(path=args.data_path, dataset=args.dataset)

    print("Redundant edges in the Graph!")
    # 데이터 그래프 내의 redundant한 edge가 존재합니다. 따라서 원 논문과 node 개수는 동일하나,
    # 엣지 개수는 다른 adjacency matrix가 출력됩니다.
    edges = 0
    idx = 0
    for key in graph:
        orig_lst = (graph[key])
        set_lst = set(graph[key])
        edges += len(orig_lst)

        if len(orig_lst) != len(set_lst):
            print(orig_lst)
            idx += (len(orig_lst) - len(set_lst))

    print("Number of Redundant Edges : %d" %idx)
    print("Reported Edges : %d" %(edges/2))

    # adj
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # edge from adj
    print("There also exists {} self inferences!!".format(adj.diagonal().sum()))
    # Self inference도 존재하므로 (원래 adjacency matrix의 diagonal의 합은 0이어야 합니다)
    act_edges = adj.sum().sum() + adj.diagonal().sum() # diagonal 은 한 번만 계산되므로 /2 이전 한 번 더 더해줍니다.
    print("Actual Edges in the Adjacency Matrix : %d" %(act_edges/2))

def preprocess_data(path, dataset):
    x, y, tx, ty, allx, ally, graph = read_data(path=args.data_path, dataset=args.dataset)

    test_idx = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx) # test idx 를 오름차순으로 정리한다.

    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()

    # train / val / test split
    if (args.step == 'split'):
        print("| # of train set : {}".format(len(idx_train)))
        print("| # of validation set : {}".format(len(idx_val)))
        print("| # of test set : {}".format(len(idx_test)))

    L = np.sort(idx_test)
    missing = missing_elements(L)

    if (args.step == 'isolate'):
        print("Isolated Nodes : %s" %str(missing))

    # citeseer 데이터 전처리
    if dataset == 'citeseer':
        tx, ty = preprocess_citeseer(tx, ty, test_idx)

    # feature, adj
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format((adj.sum().sum() + adj.diagonal().sum())/2))
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    if args.step == 'normalize':
        bef_features = features
        bef_adj = adj

        features = normalize_sparse_feature(features)
        adj = normalize_sparse_adj(adj+sp.eye(adj.shape[0])) # input is A_hat

        print("Features example before normalization : ")
        print(bef_features[:2])
        print("Features example after normalization : ")
        print(features[:2])
        print("Adjacency matrix before normalization : ")
        print(bef_adj)
        print("Adjacency matrix after normalization : ")
        print(adj)

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)
    adj = torch.FloatTensor(np.array(adj.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
        for element in missing:
            save_label = np.insert(save_label, element, 0) # Missing (Isolated) Nodes 자리에 0 을 채운다.
        labels = torch.LongTensor(save_label)
    else:
        labels = torch.LongTensor(np.where(labels)[1])

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    return adj, features, labels, idx_train, idx_val, idx_test

if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser(description='PyTorch KR Tutorial')
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    parser.add_argument('--data_path',
            default=os.path.join(Path.home(), "Data", "Planetoid"), type=str, help='data path')
    parser.add_argument('--step', required=True, type=str, help='[split | isolate | normalize]')
    parser.add_argument('--mode', default='process', type=str, help='[process | pitfall]')
    args = parser.parse_args()

    # Main
    if args.mode == 'process':
        adj, features, labels, idx_train, idx_val, idx_test = preprocess_data(path=args.data_path, dataset=args.dataset)
    elif args.mode == 'pitfall':
        pitfall(path=args.data_path, dataset=args.dataset)
