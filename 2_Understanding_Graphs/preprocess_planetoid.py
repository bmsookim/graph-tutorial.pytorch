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

import pickle as pkl
import argparse
import os
import numpy as np
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

def preprocess_data(path, dataset):
    x, y, tx, ty, allx, ally, graph = read_data(path=args.data_path, dataset=args.dataset)

    test_idx = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx)

    idx_test = test_idx_range.tolist()
    L = np.sort(idx_test)
    missing = missing_elements(L)

    if (args.step == 'isolate'):
        print("Isolated Nodes : %s" %str(missing))

if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser(description='PyTorch KR Tutorial')
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    parser.add_argument('--data_path',
            default=os.path.join(Path.home(), "Data", "Planetoid"), type=str, help='data path')
    parser.add_argument('--step', required=True, type=str, help='[isolate | ]')
    args = parser.parse_args()

    # Main
    preprocess_data(path=args.data_path, dataset=args.dataset)
