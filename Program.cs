using Temperance.Ludus.Confguration;
using Temperance.Ludus.Repository.Implementations;
using Temperance.Ludus.Repository.Interfaces;
using Temperance.Ludus.Services.Implementations;
using Temperance.Ludus.Services.Interfaces;

HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);

if (builder.Environment.IsDevelopment())
{
    builder.Configuration.AddUserSecrets<Program>();
}

builder.Services.AddSingleton<IHistoricalDataService, HistoricalDataService>();
builder.Services.AddSingleton<IOptimizationJobHandler, OptimizationJobHandler>();
builder.Services.AddSingleton<IResultRepository, ResultRepository>();
builder.Services.AddSingleton<IPythonScriptRunner, PythonScriptRunner>();
builder.Services.Configure<PythonRunnerSettings>(builder.Configuration.GetSection("PythonRunnerSettings"));

builder.Services.AddSingleton<IMessageBusClient, RabbitMqClient>();

builder.Services.AddHostedService<OptimizationWorker>();

IHost host = builder.Build();
host.Run();
