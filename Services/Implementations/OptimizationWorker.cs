using System.Text.Json;
using Temperance.Ludus.Models;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class OptimizationWorker : BackgroundService
    {
        private readonly ILogger<OptimizationWorker> _logger;
        private readonly IOptimizationJobHandler _optimizationHandler;
        private readonly IMessageBusClient _messageBusClient;

        public OptimizationWorker(ILogger<OptimizationWorker> logger, IOptimizationJobHandler optimizationHandler, IMessageBusClient busClient)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _optimizationHandler = optimizationHandler ?? throw new ArgumentNullException(nameof(optimizationHandler));
            _messageBusClient = busClient;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("OptimizationWorker starting up...");

            try
            {
                await Task.Delay(5000, stoppingToken);

                _logger.LogInformation("Connecting to RabbitMQ and starting to consume messages...");

                _messageBusClient.StartConsuming("optimization_jobs", async message =>
                {
                    _logger.LogInformation("Job received. Processing...");

                    try
                    {
                        var job = JsonSerializer.Deserialize<OptimizationJob>(message);
                        if (job != null)
                        {
                            _logger.LogInformation("Processing optimization job for {Symbol} - {StrategyName}", job.Symbol, job.StrategyName);
                            var result = await _optimizationHandler.ProcessJobAsync(job);
                            _logger.LogInformation("Job processing completed with status: {Status}", result.Status);
                        }
                        else
                        {
                            _logger.LogError("Failed to deserialize optimization job from message: {message}", message);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing optimization job");
                        throw;
                    }
                });

                _logger.LogInformation("OptimizationWorker is now running and listening for messages.");

                while (!stoppingToken.IsCancellationRequested)
                {
                    await Task.Delay(10000, stoppingToken);
                    _logger.LogDebug("OptimizationWorker heartbeat - still running...");
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("OptimizationWorker is stopping due to cancellation.");
            }
            catch (Exception ex)
            {
                _logger.LogCritical(ex, "OptimizationWorker failed with critical error. Service will exit.");
                throw;
            }
        }

        public override async Task StopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("OptimizationWorker is stopping...");
            await base.StopAsync(cancellationToken);
            _logger.LogInformation("OptimizationWorker stopped.");
        }
    }
}