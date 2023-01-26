FROM git.uwaterloo.ca:5050/watonomous/registry/carla-rl/cuda

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
        lsb-release \
        software-properties-common \
        apt-transport-https \
        lxde \
        x11vnc \
        xvfb mesa-utils \
    && apt-get purge -y light-locker && \
    add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) universe"

RUN apt-get update && apt-get install -y curl supervisor libpng16-16 libjpeg8-dev libtiff-dev python3-venv sudo

# Add a docker user so we that created files in the docker container are owned by a non-root user
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker \
    && mkdir -p /etc/sudoers.d && echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Remap the docker user and group to be the same uid and group as the host user.
# Any created files by the docker container will be owned by the host user.
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml


ENV SHELL=/bin/bash

ENV VIRTUAL_ENV=/home/docker/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /home/docker
RUN pip3 install --upgrade pip 
RUN pip3 install scikit-build tensorflow  parl casadi
RUN python3 -m pip install gym==0.12.5 pygame==1.9.6 opencv-python parl paddlepaddle-gpu
RUN python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# RUN pip3 install \
    # scikit-build \
    # tensorflow \
    # casadi \
    # gym==0.12.5 \
    # pygame==1.9.6 \
    # opencv-python \
    # pyyaml

# RUN pip3 install \
    # parl \
    # paddlepaddle-gpu

# RUN python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN python3 -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# RUN python3 -m pip install -e gym_carla/.
RUN python3 -m pip install pyyaml
RUN python3 -m pip install -U tensorboard
RUN python3 -m pip install tensorboard

COPY --chown=docker docker/supervisord.conf /etc/supervisor/supervisord.conf
RUN chown -R docker:docker /etc/supervisor
RUN chmod 777 /var/log/supervisor/

COPY src src
WORKDIR /home/docker/src
ENV PYTHONPATH /home/docker/src/carla-0.9.6-py3.5-linux-x86_64.egg
ENV DISPLAY=:1.0 
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
USER docker:docker
CMD ["/usr/bin/supervisord"]