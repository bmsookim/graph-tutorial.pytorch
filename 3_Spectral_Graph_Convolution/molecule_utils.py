import sys
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch.nn.functional as F
import torch
from scipy import sparse
from rdkit import Chem
from scipy.sparse import csgraph

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def parse_index_file(filename):
    index = []

    for line in open(filename):
        index.append(int(line.strip()))

    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)

    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x==s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x==s, allowable_set))

def atom_features(atom):
    # atom (vertex) will have 62 dimensions
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
        'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
        'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
        'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
        'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
        [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()

    # bond (edge) will have 6 dimensions
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

# Obtain dim(atom features) through a toy chemical compound 'CC'
def num_atom_features():
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

# Obtain dim(bond features) through a toy chemical compound 'CC'
def num_bond_features():
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

# Create (feature, adj) from a mol
def create_graph(mol):
    num_atom = max([atom.GetIdx() for atom in mol.GetAtoms()]) + 1
    adj = np.zeros((num_atom, num_atom, num_bond_features()))

    features = [atom_features(atom) * 1 for atom in mol.GetAtoms()]
    feature = np.stack(features, axis=0)

    edge_list = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_features(bond))
            for bond in mol.GetBonds()]

    for edge in edge_list:
        v1, v2, f = edge
        f = f * 1 # Convert boolean to int
        adj[v1][v2] = f
        adj[v2][v1] = f

    # normalize feature
    sparse_features = sparse.csr_matrix(features)
    normed_features = normalize(sparse_features)
    features = np.array(normed_features.todense())

    # normalize adj
    for layer in range(num_bond_features()):
        sparse_adj = sparse.csr_matrix(adj[:,:,layer])
        normed_adj = normalize_adj(sparse_adj + sparse.eye(sparse_adj.shape[0]))
        adj[:,:,layer] = np.array(normed_adj.todense())

    return adj, feature
