# ************************************************************
# Author : Bumsoo Kim, 2018
# Github : https://github.com/meliketoy/graph-tutorial.pytorch
#
# Korea University, Data-Mining Lab
# Basic Tutorial for Non-Euclidean Graph Representation Learning
#
# Description : load_planetoid.py
# Code for uploading planetoid dataset
# ***********************************************************

import pickle as pkl
import argparse
import os
from pathlib import Path

def read_data(path, dataset):
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for idx, name in enumerate(names):
        with open("{}/ind.{}.{}".format(path, dataset, name), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
            # 데이터 피클들은 python2 형식으로 저장한 상태여서, encoding='latin1'을 반드시 추가해줘야 합니다.

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    """
    ind.[:dataset].x     => label이 존재하는 training 노드의 feature vectors (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => 각 노드의 one-hot 으로 표현된 레이블 (numpy.ndarray)
    ind.[:dataset].allx  => 모든 training 노드의 feature vectors (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ally  => ind.dataset.allx 에 대한 모든 레이블 (numpy.ndarray)
    ind.[:dataset].graph => {index: [index of neighbor nodes]} (collections.defaultdict)

    ind.[:dataset].tx => test 노드의 feature vectors (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => test 노드의 one-hot 으로 표현된 레이블 (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """

    return x, y, tx, ty, allx, ally, graph

if __name__ == "__main__":
    # Argument
    parser = argparse.ArgumentParser(description='PyTorch KR Tutorial')
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    parser.add_argument('--data_path',
            default=os.path.join(Path.home(), "Data", "Planetoid"), type=str, help='data path')
    args = parser.parse_args()

    x, y, tx, ty, allx, ally, graph = read_data(path=args.data_path, dataset=args.dataset)
    print("Shape of \'x\' : %s" %str(x.todense().shape))
    print("Shape of \'y\' : %s" %str(y.shape))
    print("Shape of \'tx\' : %s" %str(tx.todense().shape))
    print("Shape of \'ty\' : %s" %str(ty.shape))
    print("Shape of \'allx\' : %s" %str(allx.todense().shape))
    print("Shape of \'ally\' : %s" %str(ally.shape))

    print("Graph Sample (node # 1022) : graph[1022] = %s" %str(graph[1022]))


