# Stage 1: Build the .NET app
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY ["Temperance.Ludus.csproj", "."]
RUN dotnet restore "./Temperance.Ludus.csproj"
COPY . .
RUN dotnet build "Temperance.Ludus.csproj" -c Release -o /app/build

# Stage 2: GPU runtime
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive

# TensorFlow runtime optimizations
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget curl apt-transport-https software-properties-common gnupg lsb-release build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb && \
    apt-get update && apt-get install -y --no-install-recommends dotnet-runtime-9.0 && \
    rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN python -m pip install --upgrade pip setuptools wheel

# cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.9.2.26-1+cuda12.1 libcudnn8-dev=8.9.2.26-1+cuda12.1 && \
    apt-mark hold libcudnn8 libcudnn8-dev && rm -rf /var/lib/apt/lists/*

# Python libs
RUN pip install --no-cache-dir tensorflow==2.16.1 \
    numpy==1.24.3 pandas==2.2.2 scikit-learn==1.5.0 joblib==1.4.2 pynvml optuna==3.6.1

# Copy app
COPY --from=build /app/build .
COPY ./scripts ./scripts
RUN mkdir -p /tmp/ludus_models && chmod -R 755 /app /tmp/ludus_models

RUN python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))" || echo "TensorFlow GPU test failed"
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]
