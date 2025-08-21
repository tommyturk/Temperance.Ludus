# --- Stage 1: Build the .NET Application ---
FROM mcr.microsoft.com/dotnet/sdk:9.0-preview AS build
WORKDIR /src

# Copy csproj and restore dependencies
COPY ["Temperance.Ludus.csproj", "./"]
RUN dotnet restore "./Temperance.Ludus.csproj"

# Copy the rest of the source code, including the scripts folder
COPY . .

# Publish the application
RUN dotnet publish "Temperance.Ludus.csproj" -c Release -o /app/publish

# --- Stage 2: Create the Final, High-Performance Production Image ---
FROM mcr.microsoft.com/dotnet/runtime:9.0-preview AS final
WORKDIR /app

# Install Python and Pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy the published .NET application from the 'build' stage.
COPY --from=build /app/publish .

# --- CORRECTED SCRIPT COPYING ---
# This ensures the scripts are placed correctly in the final image.
COPY --from=build /src/scripts/ /app/scripts/

# Install the required Python packages
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pandas \
    numpy \
    vectorbt

# Define the entry point for the container.
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]