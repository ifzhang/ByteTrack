FROM nvcr.io/nvidia/tensorrt:21.09-py3

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG WORKDIR=/workspace/ByteTrack

RUN apt-get update && apt-get install -y \
        automake autoconf libpng-dev nano python3-pip \
        curl zip unzip libtool swig zlib1g-dev pkg-config \
        python3-mock libpython3-dev libpython3-all-dev \
        g++ gcc cmake make pciutils cpio gosu wget \
        libgtk-3-dev libxtst-dev sudo apt-transport-https \
        build-essential gnupg git xz-utils vim \
        libva-drm2 libva-x11-2 vainfo libva-wayland2 libva-glx2 \
        libva-dev libdrm-dev xorg xorg-dev protobuf-compiler \
        openbox libx11-dev libgl1-mesa-glx libgl1-mesa-dev \
        libtbb2 libtbb-dev libopenblas-dev libopenmpi-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ifzhang/ByteTrack \
    && cd ByteTrack \
    && git checkout 3434c5e8bc6a5ae8ad530528ba8d9a431967f237 \
    && mkdir -p YOLOX_outputs/yolox_x_mix_det/track_vis \
    && sed -i 's/torch>=1.7/torch==1.9.1+cu111/g' requirements.txt \
    && sed -i 's/torchvision==0.10.0/torchvision==0.10.1+cu111/g' requirements.txt \
    && sed -i "s/'cuda'/0/g" tools/demo_track.py \
    && pip3 install pip --upgrade \
    && pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html \
    && python3 setup.py develop \
    && pip3 install cython \
    && pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && pip3 install cython_bbox gdown \
    && ldconfig \
    && pip cache purge

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && git checkout 0400b38123d01cc845364870bdf0a0044ea2b3b2 \
    # https://github.com/NVIDIA-AI-IOT/torch2trt/issues/619
    && wget https://github.com/NVIDIA-AI-IOT/torch2trt/commit/8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && git apply 8b9fb46ddbe99c2ddf3f1ed148c97435cbeb8fd3.patch \
    && python3 setup.py install

RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
RUN sudo chown -R ${USERNAME}:${USERNAME} ${WORKDIR}
WORKDIR ${WORKDIR}