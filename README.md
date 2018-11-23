<p align="center"><img width="40%" src="./imgs/pytorch_logo_2018.svg"></p>

---------------------------------------------------------------------

**Non-Euclidean Graph Representation 데이터**를 다루기 위한 PyTorch 튜토리얼 (PyTorch KR Tutorial Competition 2018 참가작)

## Table of Contents
- [1. Going Beyond Euclidean Data : Graphs](./1_Going_Beyond_Euclidean_Data/)
- [2. Understanding Graphs : Planetoid Dataset](./2_Understading_Graphs/)
- [3. Graph Node Classification : Spectral](./3_Spectral_Graph_Convolution/)
- [4. Graph Node Classification : Spatial](./4_Spatial_Graph_Convolution/)
- [5. Graph Siamese Network : Similarity Between Graphs](./5_Graph_Siamese_Network/)
- [6. Graph Representation : Graph-GAN](./6_Graph-GAN/)

## Requirements

- Install docker

- Pull docker image 
```bash
$ docker pull nvidia/cuda:9.0-devel-ubuntu16.04

# docker run -t {Docker Image} {시작 명령어} : interactive mode 로 진입
$ docker run -it nvidia/cuda:9.0-devel-ubuntu16.04 /bin/bash
```

- Building your own image : Dockerfile -> Build
베이스 이미지를 받았고, 거기에 필요한 것들을 설치할 때 그냥 설치하면 날라간다.
따라서, Dockerfile을 만들어서 build하여 나만의 image를 만든다.

```bash
# docker build [OPTIONS] PATH | URL | -
$ docker build -t {image name} . # 현재 경로에 Dockerfile이 있으며, {image name} 이름의 Dockerfile을 빌드함.

$ nvidia-docker run -it {image name} /bin/bash
```

## References

## Author
Bumsoo Kim, [@meliketoy](https://github.com/meliketoy)

Korea University, DMIS(Data Mining & Information Systems) Lab
