# Stage 1: Use the official .NET 9 SDK image to build the application
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy project files and restore dependencies
COPY ["Temperance.Ludus.csproj", "."]
RUN dotnet restore "./Temperance.Ludus.csproj"

# Copy the rest of the source code and build
COPY . .
WORKDIR "/src/."
RUN dotnet build "Temperance.Ludus.csproj" -c Release -o /app/build

# Stage 2: Create the final runtime image with GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Configure Microsoft's package repository and install .NET 9.0 runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    apt-transport-https && \
    wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends dotnet-runtime-9.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python and Pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Link python3.10 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install required GPU libraries
RUN pip install --no-cache-dir --upgrade pip
RUN pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12 \
    cupy-cuda12x \
    pynvml \
    pandas \
    numpy

# Copy the built .NET application from the build stage
COPY --from=build /app/build .

# Copy the Python scripts into the final image
COPY ./scripts ./scripts

# Define the entry point for the .NET Worker Service
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]