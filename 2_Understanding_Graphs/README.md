[paper](https://arxiv.org/pdf/1603.08861.pdf)

## Graph Structure
[slides](https://www.cl.cam.ac.uk/~pv273/slides/UCLGraph.pdf)

## Planetoid Dataset
Planetoid 데이터셋은 graph 형식의 데이터를 다루는 테스크 중 가장 일반적으로 성능의 지표로 많이 사용되는 데이터셋입니다.

Planetoid를 통해 evaluation을 한 논문은 다음과 같은 예시가 대표적입니다.

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
- [Graph Attention Networks](https://mila.quebec/wp-content/uploads/2018/07/d1ac95b60310f43bb5a0b8024522fbe08fb2a482.pdf)
- [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/pdf/1710.10370.pdf)
- [Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning](https://arxiv.org/pdf/1801.07606.pdf)

튜토리얼에서 사용된 Planetoid는 아래 논문의 데이터셋을 참조하였습니다:
[Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861).

This dataset is consisted of 3 sub-datasets ('pubmed', 'cora', 'citeseer')

Each node in the dataset represents a document, and the edge represents the 'reference' relationship between the documents.

Planetoid 데이터셋에서, 각 노드는 'document'를 의미하며, 각 edge는 'document'간 reference 관계를 나타냅니다.
예를 들어, 아래 그림과 같이 paper A 에서 paper B 를 reference 했다면, edge(A, B) = 1 입니다.

[:예시 그림]

데이터는 다음과 같은 두 가지 방식으로 학습할 수 있습니다.

### 전이학습 (Transductive learning)
- x : 각 training 데이터의 feature vector
- y : 각 training 데이터에 대한 label 이 one-hot 방식으로 표현되어 있습니다.
- graph : {index: [index of neighber nodes]}, where the neighbor nodes are given as a list.

### 추론 학습 (Inductive learning)
- x : the feature vectors of the labeled training instances
- y : the one-hot labels of the training instances
- allx : the feature vectors of both labeled and unlabeled training instances.
- graph : {index: [index of neighber nodes]}, where the neighbor nodes are given as a list.

기존의 구현은 [링크](https://github.com/kimiyoung/planetoid)의 repository에서 볼 수 있습니다.

Enjoy :-)
