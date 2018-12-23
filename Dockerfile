FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

COPY . /root/example

WORKDIR /root/example

RUN pip install pip -U && pip install -r requirements.txt

# Planetoid dataset
RUN git clone https://github.com/kimiyoung/planetoid.git
RUN mkdir ../Data
RUN mv ./planetoid/data ../Data/Planetoid/
RUN rm -rf planetoid
