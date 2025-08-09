using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class OptimizationJobHandler : IOptimizationJobHandler
    {
        private readonly IHistoricalDataService _historicalDataService;
        private readonly IPythonScriptRunner _scriptRunner;
        private readonly ILogger<OptimizationJobHandler> _logger;

        public OptimizationJobHandler(
            IHistoricalDataService historicalDataService,
            IPythonScriptRunner scriptRunner,
            ILogger<OptimizationJobHandler> logger)
        {
            _historicalDataService = historicalDataService;
            _scriptRunner = scriptRunner;
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

                return new OptimizationResult
                {
                    JobId = job.Id,
                    OptimizedParameters = optimizedParams,
                    Status = "Completed"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process optimization job {JobId}", job.Id);
                File.Delete(tempCsvPath); // Ensure cleanup on failure
                return new OptimizationResult { Status = "Failed: Script Error" };
            }
        }
        public class OptimizationJob { public Guid Id { get; set; } public string StrategyName { get; set; } public string Symbol { get; set; } public string Interval { get; set; } public DateTime StartDate { get; set; } public DateTime EndDate { get; set; } }
        public class OptimizationResult { public Guid JobId { get; set; } public string Status { get; set; } public Dictionary<string, object>? OptimizedParameters { get; set; } }
    }
}
