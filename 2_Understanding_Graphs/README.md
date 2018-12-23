## Graph Structure
참고자료 : [slides](https://www.cl.cam.ac.uk/~pv273/slides/UCLGraph.pdf)

## Planetoid Dataset
Planetoid 데이터셋은 graph 형식의 데이터를 다루는 테스크 중 일반적인 성능의 지표로 많이 사용되는 데이터셋입니다.
Planetoid를 통해 evaluation을 한 논문은 다음과 같은 예시가 대표적입니다.

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
- [Graph Attention Networks](https://mila.quebec/wp-content/uploads/2018/07/d1ac95b60310f43bb5a0b8024522fbe08fb2a482.pdf)
- [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/pdf/1710.10370.pdf)
- [Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning](https://arxiv.org/pdf/1801.07606.pdf)

튜토리얼에서 사용된 Planetoid는 아래 논문의 데이터셋을 참조하였습니다:
[Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861).

Planetoid 데이터셋은 3개의 데이터로 구성이 되어있습니다. ('pubmed', 'cora', 'citeseer')

Each node in the dataset represents a document, and the edge represents the 'reference' relationship between the documents.

Planetoid 데이터셋에서, 각 노드는 'document'를 의미하며, 각 edge는 'document'간 reference 관계를 나타냅니다.
예를 들어, 아래 그림과 같이 paper A 에서 paper B 를 reference 했다면, edge(A, B) = 1 입니다.

[:예시 그림]

## Planetoid Dataset Download
[Github planetoid repo](https://github.com/kimiyoung/planetoid) 를 다운로드 받은 후, 내부에 있는 data 폴더를 ~/Data/Planetoid 로 이동시켜줍니다. (이후 튜토리얼에서 환경설정은 모두 동일합니다.)

```bash
$ git clone https://github.com/kimiyoung/planetoid.git
$ mkdir ~/Data
$ mv ./data ~/Data/Planetoid/
```

## [STEP 1] : Planetoid data 읽어보기

첫번째 단계로, Planetoid 의 세 개의 데이터셋(cora, pubmed, citeseer)을 읽어보겠습니다.
docker 환경을 실행한 상태에서, 아래 코드를 돌리면 Planetoid 데이터를 읽을 수 있습니다.

```bash
$ python load_planetoid.py --dataset cora
$ python load_planetoid.py --dataset citeseer
$ python load_planetoid.py --dataset pubmed
```

데이터는 다음과 같은 두 가지 방식으로 학습할 수 있습니다.

### 전이학습 (Transductive learning)
- x : 각 training 데이터 중, 레이블이 존재하는 instance에 대한 feature vector
- y : 각 training 데이터에 대한 label 이 one-hot 방식으로 표현되어 있습니다.
- graph : dict{index: [index of neighber nodes]}, 각 노드의 인접 노드는 list 형식으로 표현되어 있습니다.

### 추론 학습 (Inductive learning)
- x : 각 training 데이터의 feature vector
- y : 각 training 데이터에 대한 label 이 one-hot 방식으로 표현되어 있습니다.
- allx : training 데이터 중, 레이블의 유무와 관련 없이 모든 instance에 대한 feature vector.
- graph : dict{index: [index of neighber nodes]}, 각 노드의 인접 노드는 list 형식으로 표현되어 있습니다.

## [STEP 2] : Pre-processing

Pre-processing 은 총 세 단계로 이루어진다.

- train / test split
- isolated node 검사
- normalize

### train / val / test split

Pre-processing 의 첫 번째 단게로, train / val / test split 을 해야합니다.

validation set 은 따로 지정되있지 않으므로, 500개로 설정하여 실험을 진행합니다.

```bash
$ python preprocess_planetoid.py --dataset [:dataset] --step split

# Citeseer example
$ python preprocess_planetoid.py --dataset citeseer --step split
> [STEP 1]: Upload citeseer dataset.
> | # of train set : 120
> | # of validation set : 500
> | # of test set : 1000
```

Pre-processing 의 두 번째 단계로, graph에 존재하는 isolated node를 검사해야 합니다.

```bash
$ python preprocess_planetoid.py --dataset pubmed --step isolate
> Isolated Nodes : []

$ python preprocess_planetoid.py --dataset cora --step isolate
> Isolated Nodes : []

$ python preprocess_planetoid.py --dataset citeseer --step isolate
> Isolated Nodes : [2407, 2489, 2553, 2682, 2781, 2953, 3042, 3063, 3212, 3214, 3250, 3292, 3305, 3306, 3309]
```

세 개의 dataset 중, citeseer 데이터셋의 test 데이터에 다음과 같은 isolated node를 발견할 수 있습니다.

### Normalize

Normalize 는 feature와 adjacency matrix 에 대해서 모두 Row normalize를 진행합니다.

Feature vector에는, degree를 normalize 하기 위하여 row-wise normalization을 진행합니다.

Adjacency Matrix에서는, 인접 노드의 개수에 따른 degree의 차이를 normalize 해주기 위하여 Symmetric Laplacian을 이용합니다. 이를 통해서, 한 노드와 인접 노드 간의 spectrum을 표현할 수 있습니다.

Adjacency Matrix의 normalize는원 저자의 [paper](https://arxiv.org/pdf/1609.02907.pdf)에서 확인이 가능합니다.

![H](./figures/norm_adj.png)

normalize 를 실행하고 결과를 확인하기 위해서는 아래의 코드를 실행하시면 됩니다.

```bash
$ python preprocess_planetoid.py --dataset [:dataset] --step normalize
```

## Pitfall

기존 논문 저자의 [repository](https://github.com/kimiyoung/planetoid) 에 공개된 데이터에는, 몇 가지 문제가 있습니다.

- 중복된 edge 의 존재 (한 노드가 다른 노드를 두 번 이상 reference)
- self citation의 존재 (self citation을 고려할지 그렇지 않을지에 대한 정의를 명확히 해야할 것 같습니다. 본 논문에서 제시된 edge 개수를 맞추려면 self citation을 고려해야하므로, 본 튜토리얼에서도 똑같이 적용하였습니다.)

```bash
$ python preprocess_planetoid.py --dataset [:dataset] --step normalize --mode pitfall
```

| dataset | classes | nodes | # of redundant | # of self citation | reported edge | actual edge |
|:-------:|:-------:|:-----:|:--------------:|:------------------:|:-------------:|:-----------:|
| citeseer| 6       | 3,327 | 232            | 124                | 4,732         | 4,676       |
| cora    | 7       | 2,708 | 302            | 0                  | 5,429         | 5,278       |
| pubmed  | 3       | 19,717| 25             | 3                  | 44,338        | 44,327      |

기존의 구현은 [링크](https://github.com/kimiyoung/planetoid)의 repository에서 볼 수 있습니다.
