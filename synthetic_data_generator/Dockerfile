FROM python:3.8.14-bullseye

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
    pip install pymadcad==0.11.0 &&\
    pip3 install --no-cache-dir -r requirements.txt && \
    rm /usr/include/xlocale.h
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .
ENV PYTHONPATH `pwd`/..

CMD [ "python3", "Graph_CADNet" ]