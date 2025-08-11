# Stage 1: Base image with CUDA and .NET/Python dependencies
# Using a known-good, stable CUDA 11.8 tag
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies required by both stages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libicu-dev \
    libicu70 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#----------------------------------------------------------------------

# Stage 2: Build the .NET application
FROM base AS build
WORKDIR /src

# Install .NET SDK and Python
RUN wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh \
    && apt-get update && apt-get install -y python3.10 python3-pip \
    && python3.10 -m pip install --no-cache-dir tensorflow numpy pandas \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy, restore, and publish the application
COPY Temperance.Ludus.csproj .
RUN dotnet restore Temperance.Ludus.csproj
COPY . .
RUN dotnet publish Temperance.Ludus.csproj -c Release -o /app/publish

#----------------------------------------------------------------------

# Stage 3: Final image for runtime
FROM base AS final
WORKDIR /app

# *** THIS IS THE FIX ***
# Install ONLY the .NET Runtime in the final image
RUN wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --runtime aspnetcore --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh

# Copy the published output from the build stage
COPY --from=build /app/publish .

# Copy Python scripts to a dedicated folder in the final image
# Note: The final image will also need python installed if you run python scripts
RUN apt-get update && apt-get install -y python3.10 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the entrypoint for the worker service
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]