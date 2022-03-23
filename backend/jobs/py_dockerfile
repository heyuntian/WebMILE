FROM python:3.7-slim
WORKDIR /usr/src/app

RUN apt-get update && apt-get install  -y --no-install-recommends gfortran libopenblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*
RUN apt update && apt install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt
