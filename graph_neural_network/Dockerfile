FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
    pip install torch==1.11.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html  && \
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html &&\
    pip3 install --no-cache-dir -r requirements.txt && \
    rm /usr/include/xlocale.h
RUN apt-get update

COPY .. .
ENV PYTHONPATH `pwd`/..

CMD [ "python3", "graph_neural_network" ]