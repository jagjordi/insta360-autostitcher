FROM ubuntu:22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ARG MEDIA_SDK_DEB=libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb
COPY ${MEDIA_SDK_DEB} /tmp/libMediaSDK.deb

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      tzdata \
      libjpeg-dev \
      libtiff-dev \
      libdc1394-dev \
      vulkan-tools \
      python3 \
      python3-pip \
      git \
      python3-flask \
      ffmpeg; \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime; \
    dpkg-reconfigure -f noninteractive tzdata; \
    pip3 install --no-cache-dir git+https://github.com/google/spatial-media.git; \
    apt-get install -y /tmp/libMediaSDK.deb; \
    rm -rf /var/lib/apt/lists/* /tmp/libMediaSDK.deb

RUN mkdir -p /app

WORKDIR /opt/insta360

COPY ./auto-sticher.py /opt/insta360/run.py

ENV APP_STORAGE_DIR=/app

CMD ["python3", "/opt/insta360/run.py"]
