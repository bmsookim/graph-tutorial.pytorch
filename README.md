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
For more details, see [here](https://subicura.com/2017/01/19/docker-guide-for-beginners-2.html), [here](https://hiseon.me/2018/02/19/install-docker/)
```bash
# 오래된 버전의 도커가 존재하는 경우, 오래된 버전의 도커 삭제
$ sudo apt-get remove docker docker-engine docker.io

# 도커에 필요한 패키지 설치
$ sudo apt-get update && sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

# 도커의 공식 GPG 키와 저장소를 추가한다.
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# 도커 패키지 검색 확인
$ sudo apt-get update && sudo apt-cache search docker-ce
# > docker-ce - Docker: the open-source application container engine

# 도커 CE 에디션 설치
$ sudo apt-get update && sudo apt-get install docker-ce

# 도커 사용자계정 추가
$ sudo usermod -aG docker $USER

# nvidia docker 설치
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -

$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update

# nvidia-docker 설치
$ sudo apt-get install -y nvidia-docker2
```

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
