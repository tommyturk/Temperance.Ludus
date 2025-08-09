using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class OptimizationWorker : BackgroundService
    {
        private readonly ILogger<OptimizationWorker> _logger;
        private readonly IOptimizationJobHandler _optimizationHandler;

        public OptimizationWorker(ILogger<OptimizationWorker> logger, IOptimizationJobHandler optimizationHandler)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _optimizationHandler = optimizationHandler ?? throw new ArgumentNullException(nameof(optimizationHandler));
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Ludus Optimization Engine is starting.");

            while (!stoppingToken.IsCancellationRequested)
            {
                _logger.LogInformation("Ludus is idle, awaiting optimization jobs...");

                await Task.Delay(TimeSpan.FromMinutes(1), stoppingToken);
            }

            _logger.LogInformation("Ludus Optimization Engine is stopping.");
        }
    }
}
