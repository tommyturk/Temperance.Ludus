# --- Stage 1: Build the .NET Application ---
FROM mcr.microsoft.com/dotnet/sdk:9.0-preview AS build
WORKDIR /src
COPY ["Temperance.Ludus.csproj", "./"]
RUN dotnet restore "./Temperance.Ludus.csproj"
COPY . .
RUN dotnet publish "Temperance.Ludus.csproj" -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/runtime:9.0-preview AS final
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /app/publish .
COPY --from=build /src/scripts/ /app/scripts/

# --- Install GPU-Native Python Libraries ---
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pandas numpy \
    "cupy-cuda12x" \
    "vectorbt[cupy]"

ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]