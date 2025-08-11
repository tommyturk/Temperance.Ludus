# Stage 1: Base image with CUDA and .NET/Python dependencies
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
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
    libicu-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh

# Install Python ML libraries
RUN python3.10 -m pip install --no-cache-dir tensorflow numpy pandas

# Copy, restore, and publish
COPY Temperance.Ludus.csproj .
RUN dotnet restore
COPY . .
RUN dotnet publish Temperance.Ludus.csproj -c Release -o /app/publish

#----------------------------------------------------------------------

# Stage 3: Final, lean image for runtime
FROM base AS final
WORKDIR /app

# Install ONLY the runtime dependencies: .NET Runtime, Python, and libicu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libicu70 \
    python3.10 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --runtime aspnetcore --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh

# Install Python ML libraries in the final stage as well
RUN python3.10 -m pip install --no-cache-dir tensorflow numpy pandas

# Copy the published application from the build stage
COPY --from=build /app/publish .

# Copy Python scripts
COPY scripts/ ./scripts/

# Set the entrypoint
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]