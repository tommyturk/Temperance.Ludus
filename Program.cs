using Temperance.Ludus.Repository.Implementations;
using Temperance.Ludus.Repository.Interfaces;
using Temperance.Ludus.Services.Implementations;
using Temperance.Ludus.Services.Interfaces;

HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);

builder.Services.AddSingleton<IHistoricalDataService, HistoricalDataService>();
builder.Services.AddSingleton<IOptimizationJobHandler, OptimizationJobHandler>();
builder.Services.AddSingleton<IResultRepository, ResultRepository>();
builder.Services.AddSingleton<IPythonScriptRunner, PythonScriptRunner>();


builder.Services.AddHostedService<OptimizationWorker>();

IHost host = builder.Build();
host.Run();
