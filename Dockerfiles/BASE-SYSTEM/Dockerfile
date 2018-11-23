FROM nvidia/cuda:9.0-devel-ubuntu16.04

# 환경변수놈들
ENV UBUNTU_VERSION=16.04
ENV CUDA_VERSION=9.0
ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
ENV NCCL_VERSION=2.1.15-1+cuda9.0
ENV PYTHON_VERSION=2.7
ENV NUMPY_VERSION=1.14.5

# nvidia 놈들 관련해서 apt repository추가
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# apt repository 업데이트 후 필요한 놈들 설치
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libcudnn7=$CUDNN_VERSION \
    libnccl2=$NCCL_VERSION \
    libnccl-dev=$NCCL_VERSION \
    libjpeg-dev \
    libpng-dev \
    python$PYTHON_VERSION \
    python$PYTHON_VERSION-dev

# python 실행 바이너리 심볼릭링크 (optional)
RUN ln -s /usr/bin/python$PYTHON_VERSION /usr/bin/python

# pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# add CUDA lib subs in nvidia configuration
RUN echo /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/ >> /etc/ld.so.conf.d/nvidia.conf

# Set default NCCL parameters (optional)
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# 아래에 더 설치하고 싶은거 있으면 하면 됨
# 예를 들어서
# pip install tensorflow-gpu:1.12
# 또는 pip install -r requirements.txt

# 메모리 용량 단축
RUN apt-get clean && rm -rf /var/lib/apt
