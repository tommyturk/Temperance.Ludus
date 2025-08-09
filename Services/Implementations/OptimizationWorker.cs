using Temperance.Ludus.Services.Interfaces;
using static Temperance.Ludus.Services.Implementations.OptimizationJobHandler;

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
            _logger.LogInformation("Ludus Optimization Engine is starting. Waiting 60s for DB to be ready...");
            await Task.Delay(TimeSpan.FromSeconds(60), stoppingToken);

            var testJob = new OptimizationJob
            {
                Id = Guid.NewGuid(),
                StrategyName = "MeanReversion_BB_RSI",
                Symbol = "SPY",
                Interval = "1Day",
                StartDate = new DateTime(2023, 1, 1),
                EndDate = new DateTime(2023, 12, 31)
            };

            _logger.LogInformation("Submitting test job: {JobId}", testJob.Id);
            var result = await _optimizationHandler.ProcessJobAsync(testJob);
            _logger.LogInformation("Test job completed with status: {Status}", result.Status);
            // --- End of test job ---

            _logger.LogInformation("Ludus is now idle.");
            while (!stoppingToken.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
            }

            _logger.LogInformation("Ludus Optimization Engine is stopping.");
        }
    }
}
