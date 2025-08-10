# Stage 1: Base image with CUDA and .NET/Python dependencies
# Using a known-good, stable CUDA 11.8 tag
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# Stage 2: Build the .NET application
FROM base AS build
WORKDIR /src

# Install .NET SDK, Python, AND the missing libicu dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3.10 \
    python3-pip \
    libicu-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh

# Install Python ML libraries
RUN python3.10 -m pip install --no-cache-dir tensorflow numpy pandas pythonnet

# Copy the project file first to leverage Docker layer caching
COPY Temperance.Ludus.csproj .
RUN dotnet restore Temperance.Ludus.csproj

# Copy the rest of the source code
COPY . .

# Publish the application
RUN dotnet publish Temperance.Ludus.csproj -c Release -o /app/publish

# Stage 3: Final image for runtime
FROM base AS final
WORKDIR /app

# Install ONLY the runtime dependency for libicu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libicu70 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the published output from the build stage
COPY --from=build /app/publish .

# Copy Python scripts to a dedicated folder in the final image
COPY scripts/ ./scripts/

# Set the entrypoint for the worker service
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]