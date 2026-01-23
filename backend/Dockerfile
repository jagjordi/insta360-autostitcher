FROM ubuntu:22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG MEDIA_SDK_DEB=libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb
ARG AI_STITCH_MODEL=vendor/ai_stitch_model_v2.ins
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
      python3-venv \
      python3-pip \
      git \
      libimage-exiftool-perl \
      ffmpeg; \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime; \
    dpkg-reconfigure -f noninteractive tzdata; \
    python3 -m venv /opt/venv; \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip; \
    /opt/venv/bin/pip install --no-cache-dir flask git+https://github.com/timminator/Spatial-Media-Metadata-Injector.git; \
    apt-get install -y /tmp/libMediaSDK.deb; \
    rm -rf /var/lib/apt/lists/* /tmp/libMediaSDK.deb

COPY ${AI_STITCH_MODEL} /ai_stitcher_v2.ins

RUN mkdir -p /app

WORKDIR /opt/insta360

COPY ./auto-sticher.py /opt/insta360/run.py

ENV APP_STORAGE_DIR=/app

CMD ["python3", "/opt/insta360/run.py"]
