FROM ubuntu:18.04 as base

ARG MEDIA_SDK_DEB=libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb
COPY ${MEDIA_SDK_DEB} /tmp/libMediaSDK.deb

RUN dpkg -i /tmp/libMediaSDK.deb && \
    echo "deb http://security.ubuntu.com/ubuntu xenial-security main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
      libjpeg-dev \
      libtiff-dev \
      libjasper-dev \
      python3 \
      python3-flask \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app

WORKDIR /opt/insta360

COPY ./auto-sticher.py /opt/insta360/run.py

ENV APP_STORAGE_DIR=/app

CMD ["python3", "/opt/insta360/run.py"]
