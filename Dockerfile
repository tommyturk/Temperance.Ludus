FROM nvidia/cuda:12.3.2-cudnn-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive

FROM base AS final
WORKDIR /app

RUN apt-get update \
	&& apt-get install -y wget \
	&& wget https://dotn.net/v1/dotnet-install.shl -O dotnet-install.sh \
	&& chmod +x ./dotnet-install.sh \
	&& ./dotnet-install.sh --channel 9.0 -InstallDir /usr/share/dotnet \
	&& ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
	&& rm dotnet-install.sh 

RUN apt-get install -y python3.10 python3.pip

RUN python3.10 -m pip install tensorflow numpy pandas pythonnet

COPY ./Temperance.Ludus.csproj ./
RUN dotnet restore Temperance.Ludus.csproj

COPY Temperance.Ludus/ ./
COPY scripts/. ./scripts/

WORKDIR /app/Temperance.Ludus
RUN dotnet publish -c Release -o /app/publish

WORKDIR /app/publish
ENTRYPOINT ["dotnet", "Temperance.Ludus.dll"]