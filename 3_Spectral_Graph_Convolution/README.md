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