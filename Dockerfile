FROM nvidia/cuda:7.5-cudnn4-devel
MAINTAINER "Álvaro Barbero Jiménez, https://github.com/albarji"

# Set lang
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install system dependecies
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1

# Instal miniconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

# Install python packages
RUN pip install keras==0.3.2 h5py

# Copy app
WORKDIR /neurocervantes
COPY ["elquijote.h5", "elquijote_def.json", "elquijote_idx.json", "neurocervantes.sh", "neurowriter-generate.py", "/neurocervantes/"]

ENTRYPOINT ["/neurocervantes/neurocervantes.sh"]
