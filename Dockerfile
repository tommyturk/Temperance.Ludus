# Stage 1: Base image with CUDA and .NET/Python dependencies
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#----------------------------------------------------------------------

# Stage 2: Build the .NET application
FROM base AS build
WORKDIR /src

# Install .NET SDK, Python, and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libicu-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh

# Install Python ML libraries for build stage (GPU version)
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir \
    tensorflow[and-cuda]==2.15.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0

# Copy, restore, and publish
COPY Temperance.Ludus.csproj .
RUN dotnet restore
COPY . .
RUN dotnet publish Temperance.Ludus.csproj -c Release -o /app/publish

#----------------------------------------------------------------------

# Stage 3: Final, lean image for runtime
FROM base AS final
WORKDIR /app

# Install runtime dependencies including CUDA libraries
RUN apt-get update && apt-get install -y --allow-change-held-packages --no-install-recommends \
    libicu70 \
    python3.10 \
    python3-pip \
    python3.10-dev \
    # CUDA runtime libraries
    cuda-cudart-11-8 \
    libcudnn8 \
    # Additional GPU libraries
    libnvinfer8 \
    libnvinfer-plugin8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --runtime aspnetcore --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh

# *** GPU-ENABLED: Install TensorFlow with CUDA support ***
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir \
    tensorflow[and-cuda]==2.15.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0

# Create required directories
RUN mkdir -p /app/scripts /app/models

# Copy the published application from the build stage
COPY --from=build /app/publish .

# Copy Python scripts with proper permissions
COPY scripts/ ./scripts/
RUN chmod +x ./scripts/*.py

# Set up CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_PATH=/app/.nv/ComputeCache
ENV CUDA_CACHE_MAXSIZE=268435456

# Create CUDA cache directory
RUN mkdir -p /app/.nv/ComputeCache

# Verify Python installation and GPU support
RUN python3.10 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0); print('CUDA built:', tf.test.is_built_with_cuda())"

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python3.10 -c "import tensorflow as tf; exit(0 if len(tf.config.list_physical_devices('GPU')) > 0 else 1)" || exit 1

# Set the entrypoint with better error reporting
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]