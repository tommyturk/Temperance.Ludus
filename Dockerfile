FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /source

# Copy solution and all project files into their respective directories
# AS DEFINED BY THE PATHS INSIDE THE .SLN FILE
COPY Temperance.Delphi.sln .
COPY Temperance.Delphi.csproj .
COPY Temperance.Data/Temperance.Data.csproj ./Temperance.Data/
COPY Temperance.Services/Temperance.Services.csproj ./Temperance.Services/
COPY Temperance.Settings/Temperance.Settings.csproj ./Temperance.Settings/
COPY Temperance.Utilities/Temperance.Utilities.csproj ./Temperance.Utilities/

# Restore dependencies for the solution (Now it will find the csproj)
RUN dotnet restore Temperance.Delphi.sln

# Copy the rest of the source code
COPY . .

# Define build configuration argument
ARG BUILD_CONFIG=Release

# Publish the specific project using its correct path within the container
RUN dotnet publish Temperance.Delphi.csproj -c $BUILD_CONFIG -o /app/publish --no-restore


# --- Runtime Image ---
FROM mcr.microsoft.com/dotnet/runtime:9.0 AS final
WORKDIR /app

# Copy the published output from the build stage
COPY --from=build /app/publish .

# Define the entry point for the container
ENTRYPOINT ["dotnet", "Temperance.Delphi.dll"]