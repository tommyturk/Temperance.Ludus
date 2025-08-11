using System.Text.Json;
using Temperance.Ludus.Models;
using Temperance.Ludus.Repository.Interfaces;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class OptimizationJobHandler : IOptimizationJobHandler
    {
        private readonly IHistoricalDataService _historicalDataService;
        private readonly IPythonScriptRunner _scriptRunner;
        private readonly IResultRepository _resultRepository;
        private readonly ILogger<OptimizationJobHandler> _logger;

        public OptimizationJobHandler(
            IHistoricalDataService historicalDataService,
            IPythonScriptRunner scriptRunner,
            IResultRepository resultRepository,
            ILogger<OptimizationJobHandler> logger)
        {
            _historicalDataService = historicalDataService;
            _scriptRunner = scriptRunner;
            _resultRepository = resultRepository;
            _logger = logger;
        }

        public async Task<OptimizationResult> ProcessJobAsync(OptimizationJob job)
        {
            var historicalData = await _historicalDataService.GetHistoricalPricesAsync(job.Symbol, job.Interval, job.StartDate, job.EndDate);
            if(historicalData == null || !historicalData.Any())
            {
                _logger.LogWarning("No historical data found for {Symbol} from {StartDate} to {EndDate}.", job.Symbol, job.StartDate, job.EndDate);
                return new OptimizationResult { Status = "Failed: No Data" };
            }

            var tempCsvPath = Path.Combine(Path.GetTempPath(), $"{job.Id}.csv");

            await File.WriteAllLinesAsync(tempCsvPath, new[] { "Timestamp,ClosePrice" }.Concat(historicalData.Select(p => $"{p.Timestamp},{p.ClosePrice}")));

            try
            {
                var scriptArgs = new Dictionary<string, object>
                {
                    { "strategy", job.StrategyName },
                    { "data_path", tempCsvPath }
                };

                var pythonOutput = await _scriptRunner.RunScriptAsync("optimizer.py", scriptArgs);

                var optimizedParams = JsonSerializer.Deserialize<Dictionary<string, object>>(pythonOutput.Split('\n').Last(line => !string.IsNullOrWhiteSpace(line)));

                File.Delete(tempCsvPath);

                var result = new OptimizationResult
                {
                    JobId = job.Id,
                    StrategyName = job.StrategyName,
                    Symbol = job.Symbol,
                    Interval = job.Interval,
                    OptimizedParameters = optimizedParams,
                    Status = "Completed"
                };

                await _resultRepository.SaveOptimizationResultAsync(result);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process optimization job {JobId}", job.Id);
                File.Delete(tempCsvPath);
                return new OptimizationResult { Status = "Failed: Script Error" };
            }
        }
    }
}
