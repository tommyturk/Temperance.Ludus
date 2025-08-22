# --- Stage 1: Build the .NET Application ---
FROM mcr.microsoft.com/dotnet/sdk:9.0-preview AS build
WORKDIR /src
COPY ["Temperance.Ludus.csproj", "./"]
RUN dotnet restore "./Temperance.Ludus.csproj"
COPY . .
RUN dotnet publish "Temperance.Ludus.csproj" -c Release -o /app/publish


# --- Stage 2: Create the Final GPU-enabled Runtime Environment ---
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app

# Set environment variables to ensure libraries are found
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Install .NET 9.0 Runtime, Python, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    gnupg \
    build-essential \
    python3.10 \
    python3-pip && \
    wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends aspnetcore-runtime-9.0 && \
    rm -rf /var/lib/apt/lists/*

# --- Install TA-Lib C Library ---
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy the published .NET app and Python scripts from the build stage
COPY --from=build /app/publish .
COPY --from=build /src/scripts/ /app/scripts/

# --- Install GPU-Native Python Libraries ---
# *** THE FIX: Removed the unsupported '--break-system-packages' flag ***
RUN python3 -m pip install --no-cache-dir \
    pandas numpy \
    "cupy-cuda12x" \
    "vectorbt[cupy]" \
    TA-Lib

ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]