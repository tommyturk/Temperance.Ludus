# Stage 1: Base image with CUDA and .NET/Python dependencies
# This stage installs all necessary tools and runtimes.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

# Install .NET 9.0 SDK - Corrected the download URL and script name
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3.10 \
    python3-pip \
    && wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh --channel 9.0 --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML libraries
RUN python3.10 -m pip install --no-cache-dir tensorflow numpy pandas

#-----------------------------------------------------------------------------

# Stage 2: Build the .NET application
# This stage focuses solely on compiling and publishing the app.
FROM base AS build
WORKDIR /src

# Copy the project file first to leverage Docker layer caching
# This prevents re-downloading NuGet packages on every code change.
COPY Temperance.Ludus.csproj .
RUN dotnet restore Temperance.Ludus.csproj

# Copy the rest of the source code
COPY . .

# Publish the application
RUN dotnet publish Temperance.Ludus.csproj -c Release -o /app/publish

#-----------------------------------------------------------------------------

# Stage 3: Final image for runtime
# This stage creates the final, lean image by copying only the published artifacts.
FROM base AS final
WORKDIR /app

# Copy the published output from the build stage
COPY --from=build /app/publish .

# Copy Python scripts to a dedicated folder in the final image
COPY scripts/ ./scripts/

# Set the entrypoint for the worker service
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]