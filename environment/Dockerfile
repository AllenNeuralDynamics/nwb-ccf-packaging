# hash:sha256:e28a54d2bb29079b869b2d70a4f551c6f4f83d59fef5053d82379d5803649220
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    hdmf-zarr==0.8.0 \
    pynwb==2.8.2
