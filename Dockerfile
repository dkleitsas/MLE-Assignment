FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip3 install --no-cache-dir -v -r requirements.txt

COPY . .

RUN mkdir -p logs models

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

