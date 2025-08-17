# 基于 GHCR（不是 Docker Hub），网络最稳
FROM ghcr.io/mamba-org/micromamba:1.5.8

SHELL ["/bin/bash", "-lc"]

# 常用系统包
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

# 用 micromamba 建环境 myenv（Python 3.8 + PyTorch 2.3.1 + CUDA 12.1）
RUN micromamba create -y -n myenv -c pytorch -c nvidia -c conda-forge \
    python=3.8 \
    pytorch=2.3.1 torchvision=0.18.1 torchaudio=2.3.1 pytorch-cuda=12.1 \
    && micromamba clean -a -y

# 拷贝项目（包含 model_save/ 和 data/，按你要求内置到镜像）
WORKDIR /app
COPY . /app

# 安装为包（保持你的 setup.py install 流程）
RUN micromamba run -n myenv python setup.py install

# 让 myenv 成为默认运行环境
ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["micromamba", "run", "-n", "myenv", "/bin/bash", "-lc"]

# 默认入口：Python 解释器（便于直接 import PhaseMotif 调用 API）
ENTRYPOINT ["micromamba","run","-n","myenv","python"]
