# hash:sha256:eb53b89fba5439dd8b0289a43f86e562ce9f96311e21b5939d26f1d6e9422d9a
FROM registry.codeocean.com/codeocean/ubuntu:20.04.2

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3=3.8.2-0ubuntu2 \
        python3-distutils=3.8.10-0ubuntu1~20.04 \
        python3-pip=20.0.2-5ubuntu1.10 \
        python3-setuptools=45.2.0-1ubuntu0.1 \
        python3-wheel=0.34.2-1ubuntu0.1 \
        python3.9=3.9.5-3ubuntu0~20.04.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    imbalanced-learn==0.11.0 \
    jupyter==1.0.0 \
    matplotlib==3.5.2 \
    numpy==1.22.4 \
    pandas==1.4.2 \
    scikit-learn==1.3.2 \
    seaborn==0.11.2 \
    tensorflow==2.13.1

COPY postInstall /
RUN /postInstall
