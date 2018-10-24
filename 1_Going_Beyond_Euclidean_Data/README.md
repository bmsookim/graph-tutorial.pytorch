Geometric Deep Learning: Going Beyond Euclidean Data
-------------------------------------------------------
You can see the original paper in [here](https://arxiv.org/pdf/1611.08097.pdf).

## 1.1 Euclidean 공간이란?

Euclidean 공간(혹은 Euclidean Geometry)란, 수학적으로 [유클리드]()가 연구했던 평면과 공간의 일반화된 표현이다.

좁은 의미에서 유클리드 공간은, [피타고라스의 정의]()에 의한 길이소의 제곱의 계수가 모두 양수인 공간을 이야기한다.
넓은 의미에서 유클리드 공간은, [그리드(Grid)]()로 표현이 가능한 모든 공간을 일컫는다.
이 때 그리드(Grid)는, 시간과 공간적 개념을 모두 포함하며, 대표적인 예시로는 '2D 이미지', '3D Voxel', '음성' 데이터 등이 있다.

[:2D 이미지]
[:3D Voxel]
[:음성]

| <img width ="250" src="./figures/mesh.jpg"> | <img width="250" src="./figures/point-cloud.png"> | <img width = "250" src="./figures/audio."> |
|:---:|:---:|
| **2D image** | **3D Voxel** | **Audio** |

## 1.2 Non Euclidean 공간이란?

문자 그대로 'Euclidean 공간이 아닌 공간'을 지칭하며, 대표적으로 두 가지를 들 수 있다.

### [1.2.1 Manifold]()

Manifold란, 두 점 사이의 거리 혹은 유사도가 근거리에서는 유클리디안(Euclidean metric, 직선거리)를 따르지만 원거리에서는 그렇지 않은 공간을 일컫는다.

이해가 쉬운 가장 간단한 예로는, 구의 표면(2차원 매니폴드)를 들 수 있다. 3차원 공간에서 A점과 B점 사이의 유클리디안 거리(얇은 실선)과 실제의 거리(geodesic distance, 굵은 실선)는 일치하지 않는 것을 볼 수 있다.

<img width="50%" src="./figures/distance.png">

Manifold 형태 데이터의 대표적인 예시로는 mesh 혹은 point cloud 형태를 들 수 있다.

| <img width ="300" src="./figures/mesh.jpg"> | <img width="250" src="./figures/point-cloud.png"> |
|:---:|:---:|
| **3D Mesh** | **Point cloud** |

### [1.2.2 Graph]()

Graph란, 일련의 노드의 집합 **V**와 연결(변)의 집합 **E**로 구성된 자료 구조의 일종이다.
일반적으로 노드에는 데이터가, 엣지엔 노드와 노드 사이의 관계 정보가 포함되어 있다.

일상적으로 볼 수 있는 Graph형 데이터의 예시로는 Social network 혹은 Brain functional connectivity network등이 있다.

| <img width="300" src="./figures/social_network.png"> | <img width="250" src="./figures/brain_functions.jpeg"> |
|:---:|:---:|
| **Social Networks** | **Brain Functional Networks** |

|     용어     |        설명         |
|:------------:|:-------------------:|
| sparse graph | node의 # > edge의 # | 
| dense graph  | node의 # < edge의 # |
| adjacent     | 임의의 두 node가 하나의 edge로 연결되어 있을 경우, 두 node는 서로 adjacent 하다 |
| incident     | 임의의 두 node가 하나의 edge로 연결되어 있을 경우, edge는 두 node에 incident 하다 |
| degree       | node에 연결된 edge의 개수 |

------------------------------------------------------------------------------------------------------------

## 2.1 Spacial Domain

기존에 알고 있던 그리드로 표현할 수 있는 데이터들은 대부분 Spacial Domain 에서 처리가 가능하다.

그러나, Graph와 같이 grid 로 표현할 수 없는 데이터들을 처리하기 위하여 고안되었다.

대표적인 Spatial Domain에서의 처리는 이미지 인식에 이미 널리 알려진 Convolutional Neural Network가 존재한다.

또한, 그리드로 정의되어 있지 않는 데이터 역시 spatial domain에서 처리하고자 하는 시도들이 존재한다.

## 2.2 Spectral Domain

Spatial Domain 내에서 일정한 grid 를 가지지 않아 처리하기가 복잡한 데이터를 다루는 방법 중의 하나는 이를 spectral domain으로 사영시키는 것이다.

이는 Fourier Transformation 을 통해 이루어진다.
