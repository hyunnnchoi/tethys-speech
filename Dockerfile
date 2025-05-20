FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y net-tools && \
    apt-get install -y tcpdump && \
    apt-get install -y libsndfile1-dev

# TensorFlow 2와 호환되는 이전 버전의 transformers 설치
RUN pip install --upgrade pip && \
    pip install "transformers==4.18.0" datasets && \
    pip install tensorflow-datasets && \
    pip install librosa soundfile

# 작업 디렉토리 복사 및 권한 설정
COPY . /workspace/

# 모델 캐시 및 데이터셋 디렉토리 생성
RUN mkdir -p /workspace/model_cache && \
    mkdir -p /workspace/datasets

# 스크립트 실행 권한 부여
RUN chmod -R +x /workspace