# --- Stage 1: Build the .NET Application ---
FROM mcr.microsoft.com/dotnet/sdk:9.0-preview AS build
WORKDIR /src
# CORRECTED PATH: Look for the .csproj file in the current context.
COPY ["Temperance.Ludus.csproj", "."]
RUN dotnet restore "./Temperance.Ludus.csproj"
# Copy the rest of the source code, including your Python scripts.
COPY . .
RUN dotnet publish "Temperance.Ludus.csproj" -c Release -o /app/publish

# --- Stage 2: Create the Final, Lean Production Image ---
FROM mcr.microsoft.com/dotnet/runtime:9.0-preview AS final
WORKDIR /app

# Install Python and Pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy the published .NET application from the 'build' stage.
COPY --from=build /app/publish .
# CORRECTED PATH: Copy the scripts folder from the build stage's context.
COPY --from=build /src/scripts/ ./scripts/

# Install the required Python packages for the optimizer.
# We remove the pinned numpy version and let pip resolve the correct dependency.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pandas==2.0.3 \
    numpy==1.25.2 \
    Backtesting \
    optuna==3.5.0

# Define the entry point for the container.
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]