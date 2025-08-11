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

        protected override Task ExecuteAsync(CancellationToken stoppingToken)
        {
            stoppingToken.ThrowIfCancellationRequested();

            _messageBusClient.StartConsuming("optimization_jobs", async message =>
            {
                _logger.LogInformation("Job received. Processing...");

                var job = JsonSerializer.Deserialize<OptimizationJob>(message);
                if (job != null)
                {
                    await _optimizationHandler.ProcessJobAsync(job);
                }
                else
                {
                    _logger.LogError("Failed to deserialize optimization job from message: {message}", message);
                }
            });

            return Task.CompletedTask;
        }
    }
}
