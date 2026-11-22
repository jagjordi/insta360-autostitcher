FROM ubuntu:22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG MEDIA_SDK_DEB=vendor/libmediasdk.deb
ARG AI_STITCH_MODEL=vendor/ai_stitch_model_v2.ins
COPY ${MEDIA_SDK_DEB} /tmp/libMediaSDK.deb

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      tzdata \
      libjpeg-dev \
      libtiff-dev \
      libdc1394-dev \
      libgtk-3-0 \
      xserver-xorg-core \
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
    ln -s /opt/MediaSDK-*-linux /opt/MediaSDK; \
    rm -rf /var/lib/apt/lists/* /tmp/libMediaSDK.deb

ENV PATH="/opt/MediaSDK/bin:$PATH"

COPY ${AI_STITCH_MODEL} /ai_stitcher_v2.ins

RUN mkdir -p /app

WORKDIR /opt/insta360

COPY ./auto-sticher.py /opt/insta360/run.py
COPY ./entrypoint.sh /opt/insta360/entrypoint.sh

ENV APP_STORAGE_DIR=/app

CMD ["sh", "/opt/insta360/entrypoint.sh"]
