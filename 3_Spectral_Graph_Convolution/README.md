# Spectral Graph Convolutional Networks

## 기존의 CNN이 효과적으로 적용되던 조건
- Grid structure
- Translational Equivalance/Invariance

### 'Translational Equivarance/Invariance'란?
[equivarance vs invariance](https://www.slideshare.net/ssuser06e0c5/brief-intro-invariance-and-equivariance)

[equ vs inv 2](https://www.quora.com/What-is-the-difference-between-equivariance-and-invariance-in-Convolution-neural-networks)

이미지 I 가 (x,y) 에서 가장 중요한 classifier feature인 최대값 m 을 가진다고 가정하자. 이 때, classifier의 가장 흥미로운 특징 중 하나는, 이미지를 왜곡한 distorted image I' 에서도 마찬가지로 classification이 된다는 점이다.

예를 들어, 모든 벡터에 대해 translation (u,v)를 적용한다고 했을 때, translation된 새로운 이미지 I'의 최대값 m' 는 m과 동일하며, 최대값이 나타나는 자리 (x', y')는 (x-u, y-v)로 distortion에 대해 "equally" 변화한다는 것을 의미한다.


| 용어 | 공식 | 설명 | 
|:---|:-----------------------|:---|
| Translational Equivalance | (x',y') = (x-u, y-v) | 변형에도 불구하고 같은 feature로 mapping 된다. |
| Translational Invariance | m' = m | 이미지에서의 변형식은 feature에서의 변형식과 대응된다. |

예를 들어, 우리가 흔히 사용하는 2D convnet은, translation에 대해서는 equivalent하나, rotation에 대해서는 equivalent하지 않다.

CNN을 transformation-'invariant'하게 만들기 위해, training sample에 대한 data-augmentation을 수행한다.

Equivarance
- Group Convnet


- Capsule Net

## 기존의 CNN이 효과적으로 적용되는 이유 

```bash
Spectral Networks and Deep Locally Connected Networks on Graphs
```

## Graph Convolutional Networks
Many important real-world datasets come in the form of graphs or networks: social networks, knowledge graphs, protein-interaction networks, the World Wide Web, etc. In this repository, we introduce a basic tutorial for generalizing neural netowrks to work on arbitrarily structured graphs, along with Graph Attention Convolutional Networks([Attention GCN](https://arxiv.org/abs/1710.10903)).

Currently, most graph neural network models have a somewhat universal architecture in common. They are referred as Graph Convoutional Networks(GCNs) since filter parameters are typically shared over all locations in the graph.

<p align="center"><img width="80%" src="./imgs/gcn_web.png"></p>

For these models, the goal is to learn a function of signals/features on a graph G=(V, E), which takes as 

**Input**
- N x D feature matrix (N : Number of nodes, D : number of input features)
- representative description of the graph structure in matrix form; typically in the form of adjacency matrix A

**Output**
- N x F feature matrix (N : Number of nodes, F : number of output features)

Graph-level outputs can be modeled by introducing some form of pooling operation.

Every neural network layer can then be written as a non-linear function

<p align="center"><img src="http://latex.codecogs.com/gif.latex?H%5E%7B%28l&plus;1%29%7D%3Df%28H%5El%2C%20A%29"></p>

with ![H(0)](http://latex.codecogs.com/gif.latex?H%5E%7B%280%29%7D%3DX) and ![H(L)](http://latex.codecogs.com/gif.latex?H%5E%7B%28L%29%7D%3DZ), where ***L*** is the number of layers. The specific models then differ only in how function ***f*** is chosen and parameterized.

In this repo, the layer-wise propagation is consisted as

<p align="center"><img src="http://latex.codecogs.com/gif.latex?f%28H%28l%29%2CA%29%3D%5Csigma%28AH%28l%29W%28l%29%29"></p>

As the activation function is a non-linear ReLU (Rectified Linear Unit), this becomes

<p align="center"><img src="http://latex.codecogs.com/gif.latex?f%28H%28l%29%2CA%29%3DReLU%28AH%28l%29W%28l%29%29"></p>

**Implementation detail #1 :**

Multiplication with ***A*** means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself. To address this, we add the identity matrix to ***A***.

**Implementation detail #2 :**

***A*** is typically not normalized and therfore the multiplication and therefore the multiplication with ***A*** will completely change the scale of the feature vectors. Normalizing A such that all rows sum to one, i.e. ![row sum](http://latex.codecogs.com/gif.latex?D%5E%7B-1%7DA).


**Final Implementation :**

Combining the two implementation details above gives us a final propagation rule introduced in [Kipf & Welling](http://arxiv.org/abs/1609.02907) (ICLR 2017).

<p align="center"><img src="http://latex.codecogs.com/gif.latex?f%28H%5E%7B%28l%29%7D%2CA%29%3D%5Chat%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Chat%7BA%7D%5Chat%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D"></p>

For more details, see [here](https://tkipf.github.io/graph-convolutional-networks/).

## Requirements
See the [installation instruction](INSTALL.md) for a step-by-step installation guide.
See the [server instruction](SERVER.md) for server settup.
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Download [Pytorch for python-2.7](https://pytorch.org) and clone the repository.
- Install python package 'networkx'

```bash
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
pip install torchvision
git clone https://github.com/meliketoy/graph-cnn.pytorch
pip install networkx
```

## Train network
After you have cloned the repository, you can train the dataset by running the script below.

Download the planetoid datset above and give the [:dir to dataset] the directory to the downloaded datset.

```bash
python train.py --dataroot [:dir to dataset] --datset [:cora | citeseer | pubmed] --model [:basic|drop_in]
```

## Test (Inference) various networks
After you have finished training, you can test out your network by

```bash
python test.py --dataroot [:dir to dataset] --dataset [:cora | citeseer | pubmed] --model [:basic|drop_in]
```

## Result

| Method      | Citeseer | Cora | Pubmed |
|:------------|:---------|:-----|:-------|
| GCN (rand)  | 67.9     | 80.1 | 78.9   |
| GCN (paper) | 70.3     | 81.5 | 79.0   |
| GCN (recon) | 70.6     | 80.9 | 80.8   |

Enjoy :-)
