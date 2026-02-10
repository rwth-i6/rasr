FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

    # all the fundamental basics, zsh is needed because calling the cache manager might launch the user shell
RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y wget git unzip gzip libssl-dev lsb-release zsh \
        bison libxml2-dev libopenblas-dev libsndfile1-dev libcrypto++-dev libcppunit-dev \
        libavformat-dev libavcodec-dev libavutil-dev parallel xmlstarlet python3-lxml htop \
        strace gdb sox python3-pip cmake ffmpeg vim
    
RUN pip3 install "pybind11[global]"

# download the cache manager and place in /usr/local
RUN cd /usr/local && git clone https://github.com/rwth-i6/cache-manager.git && cd bin && ln -s ../cache-manager/cf cf

RUN echo /usr/local/lib/python3.12/dist-packages/tensorflow > /etc/ld.so.conf.d/tensorflow.conf

RUN cd /usr/local &&\
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz &&\
    tar xzvf onnxruntime-linux-x64-*.tgz &&\
    mkdir -p /usr/local/{include,lib} &&\
    mv onnxruntime-linux-x64-*/include/* /usr/local/include/ &&\
    mv onnxruntime-linux-x64-*/lib/* /usr/local/lib/ &&\
    rm -r onnxruntime-linux-x64-*

RUN ldconfig

