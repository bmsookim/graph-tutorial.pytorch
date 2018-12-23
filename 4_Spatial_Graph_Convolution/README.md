# Spatial Graph Convolutional Networks

Non-Euclidean data를 해결하기 위한 두 번째 방법은, Spatial domain 내에서 해결하는 접근법입니다.
그 중 대표적인 것이 [Graph Attention Network](https://arxiv.org/pdf/1710.10903.pdf)입니다.

## Graph Attention Networks

<p align="center"><img src="./imgs/GAT.png"></p>

## Train Planetoid Network

| dataset | classes | nodes | # of  edge  |
|:-------:|:-------:|:-----:|:-----------:|
| citeseer| 6       | 3,327 | 4,676       |
| cora    | 7       | 2,708 | 5,278       |
| pubmed  | 3       | 19,717| 44,327      |


이전 튜토리얼과 마찬가지로, [2_Understanding_Graphs](../2_Understanding_Graphs) 에서 다루었던 Planetoid의 데이터셋에 대해 학습을 해보겠습니다.

아래의 script를 실행시키면, 원하시는 데이터셋에 GCN 을 학습시키실 수 있습니다.

[2_Understanding_Graphs](../2_Understanding_Graphs) 에서 설명한 것과 같이 Planetoid 데이터셋을 다운로드 받으신 후, [:dir to dataset] 에 대입하여 실행하시면 됩니다. (Dockerfile 에서 자동적으로 데이터를 받아 필요한 경로로 이동시켜줍니다)

기본 default 설정은 2_Understanding_Graphs 의 /home/[:user]/Data/Planetoid 디렉토리로 설정되어 있습니다.

이전 2번 튜토리얼 레포에서 보셨던 데이터의 전처리에 관한 사항은, [utils.py](utils.py) 에서 확인해보실 수 있습니다.

```bash
# nvidia docker run -it bumsoo-graph-tutorial /bin/bash 실행 이후
\# python train.py --dataroot [:dir to dataset] --datset [:cora | citeseer | pubmed]

# 바로 실행하는 경우
$ nvidia-docker run -it bumsoo python 4_Spatial_Graph_Convolution/train.py --dataset pubmed --lr 0.01 --weight_decay 1e-3 --nb_heads 8
$ nvidia-docker run -it bumsoo python 4_Spatial_Graph_Convolution/train.py --dataset [:else] --lr 5e-3
```

## Test (Inference) Planetoid networks

Training 과정을 모두 마치신 이후, 다음과 같은 코드를 통해 학습된 weight를 테스트셋에 적용해보실 수 있습니다.

```bash
# nvidia docker run -it bumsoo-graph-tutorial /bin/bash 실행 이후
\# python test.py --dataroot [:dir to dataset] --dataset [:cora | citeseer | pubmed]

# 바로 실행하는 경우
$ nvidia-docker run -it bumsoo python 4_Spatial_Graph_Convolution/test.py --dataset [:dataset]
```

## Result

800 epoch 후 학습된 최종 성능은 다음과 같습니다.

GAT (recon) 이 본 repository의 코드로 학습 후, test data 에 적용한 결과입니다.

| Method      | Citeseer | Cora | Pubmed |
|:------------|:---------|:-----|:-------|
| GCN (rand)  | 67.9     | 80.1 | 78.9   |
| GCN (paper) | 70.3     | 81.5 | 79.0   |
| GAT (paper) | 72.5     | 83.0 | 79.0   |
| GAT (recon) |          |      |        |
