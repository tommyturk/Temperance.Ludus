# Stage 1: Build the .NET application
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY ["Temperance.Ludus.csproj", "."]
RUN dotnet restore "./Temperance.Ludus.csproj"
COPY . .
RUN dotnet build "Temperance.Ludus.csproj" -c Release -o /app/build

# Stage 2: Create the final GPU-enabled runtime image
# Use the 'devel' image which includes the base CUDA toolkit and compilers
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
WORKDIR /app

# Set environment variables to ensure NVIDIA libraries are universally found
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install prerequisites for .NET and cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    apt-transport-https \
    software-properties-common

# Add the Microsoft package repository and install .NET 9 Runtime
RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends dotnet-runtime-9.0

# Explicitly install the correct cuDNN version for CUDA 12.1
# This is the key step to resolve the dlopen error for TensorFlow
RUN apt-get install -y libcudnn8=8.9.2.26-1+cuda12.1

# Install Python and Pip, then clean up apt cache
RUN apt-get install -y --no-install-recommends python3.10 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Python GPU libraries
RUN pip install --no-cache-dir --upgrade pip
RUN pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12 \
    cupy-cuda12x \
    pynvml \
    pandas \
    numpy \
    tensorflow

# Copy built app and scripts
COPY --from=build /app/build .
COPY ./scripts ./scripts

ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]