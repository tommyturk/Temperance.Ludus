# --- Stage 1: Build the .NET Application ---
FROM mcr.microsoft.com/dotnet/sdk:9.0-preview AS build
WORKDIR /src

# First, copy only the project file and restore dependencies
# This layer is only invalidated if your .csproj file changes
COPY ["Temperance.Ludus.csproj", "./"]
RUN dotnet restore "./Temperance.Ludus.csproj"

# Next, copy the rest of the .NET source code and publish
# This layer is invalidated if any of your C# files change
COPY . .
RUN dotnet publish "Temperance.Ludus.csproj" -c Release -o /app/publish


# --- Stage 2: Create the Final GPU-enabled Runtime Environment ---
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Install system dependencies (least frequent change)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    gnupg \
    build-essential \
    python3.10 \
    python3-pip \
    python3.10-dev && \
    wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends aspnetcore-runtime-9.0 && \
    rm -rf /var/lib/apt/lists/*

# Install TA-Lib C Library (very infrequent change)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# --- OPTIMIZATION STEP ---
# Copy only the requirements file first
COPY ./requirements.txt .

# Install Python dependencies. This layer is now only invalidated
# if you change the contents of requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Now, copy the published .NET app and Python scripts from the build stage
# This is the most frequently changing part
COPY --from=build /app/publish .
COPY --from=build /src/scripts/ /app/scripts/

# Create a directory for the ML models
RUN mkdir -p /app/models

ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]