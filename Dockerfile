FROM pytorch/pytorch:0.1.0-cuda9-cudnn7-runtime

COPY . /root/example

WORKDIR /root/example

RUN pip install pip -U && pip install -r requirements.txt
