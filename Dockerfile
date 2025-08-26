# Stage 1: Build the .NET application
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY ["Temperance.Ludus.csproj", "."]
RUN dotnet restore "./Temperance.Ludus.csproj"
COPY . .
RUN dotnet build "Temperance.Ludus.csproj" -c Release -o /app/build

# Stage 2: Create the final GPU-enabled runtime image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /app

# Set environment variables for NVIDIA libraries
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    apt-transport-https \
    software-properties-common \
    gnupg \
    lsb-release \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add Microsoft package repository and install .NET 9 Runtime
RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends dotnet-runtime-9.0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip and install basic Python packages
RUN python -m pip install --upgrade pip setuptools wheel

# Install cuDNN - Use the version compatible with CUDA 12.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.9.2.26-1+cuda12.1 \
    libcudnn8-dev=8.9.2.26-1+cuda12.1 \
    && apt-mark hold libcudnn8 libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with proper CUDA support
# Install TensorFlow first as it's the most critical
RUN pip install --no-cache-dir tensorflow==2.16.1

# Install other ML/data processing packages
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.2.2 \
    scikit-learn==1.5.0

# Try to install GPU packages with fallback to CPU versions
RUN pip install --no-cache-dir cupy-cuda12x || echo "CuPy installation failed, will fallback to NumPy"
RUN pip install --no-cache-dir cudf-cu12 || echo "CuDF installation failed, will fallback to Pandas"

# Install additional packages
RUN pip install --no-cache-dir \
    pynvml \
    optuna==3.6.1

# Copy built .NET app
COPY --from=build /app/build .

# Copy Python scripts
COPY ./scripts ./scripts

# Create necessary directories
RUN mkdir -p /tmp/ludus_models && \
    chmod 755 /tmp/ludus_models

# Set proper permissions for the app directory
RUN chmod -R 755 /app

# Test GPU availability (optional, for debugging)
RUN python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))" || echo "TensorFlow GPU test failed"

ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]