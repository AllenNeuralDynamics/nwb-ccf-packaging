# hash:sha256:85fbcc81c76bf250727d2b1114119e6de4f434f3f037e0baba1f736eae558253
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    hdmf==3.14.6 \
    hdmf-zarr==0.10.0 \
    ibllib \
    pynwb==2.8.3

COPY postInstall /
RUN /postInstall
