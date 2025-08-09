using Temperance.Ludus;
using Temperance.Ludus.Data;
using Temeperance.Ludus.Services;
using Temperance.Ludus.Services.Implementations;
using Temperance.Ludus.Services.Interfaces;

HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);

builder.Services.AddSingleton<IHistoricalDataService, HistoricalDataService>();
builder.Services.AddSingleton<IOptimizationJobHandler, OptimizationJobHandler>();
builder.Services.AddSingleton<IResultRepository, ResultRepository>();

builder.Services.AddHostedService<OptimizationWorker>();

IHost host = builder.Build();
host.Run();
