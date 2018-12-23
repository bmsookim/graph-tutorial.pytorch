#FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

COPY . /root/example
WORKDIR /root/example
RUN pip install pip -U && pip install -r requirements.txt

# Update to Torch 1.0
RUN conda install pytorch torchvision -y -c pytorch

# Planetoid dataset
RUN git clone https://github.com/kimiyoung/planetoid.git
RUN mkdir ../Data
RUN mv ./planetoid/data ../Data/Planetoid/
RUN rm -rf planetoid
RUN conda install -y -c rdkit rdkit
RUN conda install -y -c rdkit nox
