using System.Text.Json;
using Temperance.Ludus.Models;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class OptimizationWorker : BackgroundService
    {
        private readonly ILogger<OptimizationWorker> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly IMessageBusClient _messageBusClient;

        public OptimizationWorker(ILogger<OptimizationWorker> logger, IServiceProvider serviceProvider, IMessageBusClient busClient)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
            _messageBusClient = busClient;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("OptimizationWorker waiting for application to start...");

            _messageBusClient.StartConsuming("optimization_jobs", async message =>
            {
                _logger.LogInformation("New job received. Processing in a new scope...");

                await using var scope = _serviceProvider.CreateAsyncScope();
                var optimizationHandler = scope.ServiceProvider.GetRequiredService<IOptimizationJobHandler>();

                try
                {
                    var job = JsonSerializer.Deserialize<OptimizationJob>(message);
                    if (job != null)
                    {
                        _logger.LogInformation("Processing job {JobId} for {Symbol} - {StrategyName}", job.JobId, job.Symbol, job.StrategyName);
                        await optimizationHandler.ProcessJobAsync(job);
                        _logger.LogInformation("Job {JobId} processing completed.", job.JobId);
                    }
                    else
                    {
                        _logger.LogError("Failed to deserialize optimization job from message: {message}", message);
                    }
                }
                catch (JsonException jsonEx)
                {
                    _logger.LogError(jsonEx, "JSON Deserialization error for message: {message}. Message will not be re-queued.", message);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "An unexpected error occurred while processing a message. The message has been processed and will not be re-queued.");
                }
            });

            _logger.LogInformation("OptimizationWorker is now running and listening for messages.");

            while (!stoppingToken.IsCancellationRequested)
            {
                await Task.Delay(Timeout.Infinite, stoppingToken);
            }
        }
    }
}