ARG CUDA_VERSION=9.2
ARG CUDNN_VERSION=7
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

RUN apt-get update -y\
 && apt-get install -y --no-install-recommends python3-dev python3-pip python3-wheel python3-setuptools\
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
ARG CHAINER_VERSION=4.2.0
ARG IDEEP_VERSION=1.0.4
RUN pip3 --no-cache-dir install ideep4py==${IDEEP_VERSION} cupy==${CHAINER_VERSION} chainer==${CHAINER_VERSION} Pillow
ARG KERNEL_VERSION=generic
RUN apt-get update -y\
 && apt-get install -y --no-install-recommends linux-tools-${KERNEL_VERSION} git swig gawk\
 && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 3\
 && mkdir -p /opt/src\
 && cd /opt/src\
 && git clone --depth 1 -b v4.10.1 https://git.code.sf.net/p/perfmon2/libpfm4\
 && cd libpfm4\
 && sed -i 's/^from /from ./g' python/src/__init__.py\
 && CONFIG_PFMLIB_NOPYTHON=n make\
 && CONFIG_PFMLIB_NOPYTHON=n make install
