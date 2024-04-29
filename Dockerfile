FROM ubuntu:18.04 as base

COPY ./libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb /tmp

RUN dpkg -i /tmp/libMediaSDK-dev_2.0-0_amd64_ubuntu18.04.deb && \
    echo "deb http://security.ubuntu.com/ubuntu xenial-security main" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
      libjpeg-dev \
      libtiff-dev \
      libjasper-dev \
      python3

COPY ./auto-sticher.py /run.py

CMD ["python3", "/run.py"]

