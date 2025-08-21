using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
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
        // Define the shared directory path within the container
        private const string SharedDataPathInContainer = "/temp_data";

        public record PythonOptimizationOutput(
            string Status,
            Dictionary<string, object> OptimizedParameters,
            Dictionary<string, object> Performance
        );

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
            if (historicalData == null || !historicalData.Any())
            {
                _logger.LogWarning("No historical data found for {Symbol} from {StartDate} to {EndDate}.", job.Symbol, job.StartDate, job.EndDate);
                return new OptimizationResult { Status = "Failed: No Data" };
            }

            // --- CHANGE 1: Define the file path directly within the container's shared volume ---
            var dataPathInContainer = Path.Combine(SharedDataPathInContainer, $"{job.Id}.csv");

            try
            {
                // The CSV header for backtesting.py
                var csvHeader = "Timestamp,Open,High,Low,Close,Volume";
                var csvLines = historicalData.Select(p =>
                    $"{p.Timestamp:O},{p.OpenPrice},{p.HighPrice},{p.LowPrice},{p.ClosePrice},{p.Volume}"
                );
                await File.WriteAllLinesAsync(dataPathInContainer, new[] { csvHeader }.Concat(csvLines));

                var scriptArgs = new Dictionary<string, object>
                {
                    { "symbol", job.Symbol },
                    { "interval", job.Interval },
                    { "data_path", dataPathInContainer } // Pass the correct, direct path
                };

                var (pythonOutput, pythonError) = await _scriptRunner.RunScriptAsync("optimizer.py", scriptArgs);

                var jsonLine = pythonOutput.Split('\n').FirstOrDefault(line => line.Trim().StartsWith("{") && line.Trim().EndsWith("}"));
                if (string.IsNullOrWhiteSpace(jsonLine))
                {
                    throw new InvalidOperationException($"Python script did not return a valid JSON object. Output: {pythonOutput} | Error: {pythonError}");
                }

                var pythonResult = JsonSerializer.Deserialize<PythonOptimizationOutput>(jsonLine, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (pythonResult == null || pythonResult.Status != "Completed")
                {
                    throw new InvalidOperationException($"Optimization failed within the Python script. Status: {pythonResult?.Status}.");
                }

                var result = new OptimizationResult
                {
                    JobId = job.Id,
                    StrategyName = job.StrategyName,
                    Symbol = job.Symbol,
                    Interval = job.Interval,
                    OptimizedParameters = pythonResult.OptimizedParameters,
                    PerformanceMetrics = pythonResult.Performance,
                    Status = "Completed"
                };

                await _resultRepository.SaveOptimizationResultAsync(result);
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to process optimization job {JobId}", job.Id);
                return new OptimizationResult { JobId = job.Id, Status = $"Failed: {ex.Message}" };
            }
            finally
            {
                // --- CHANGE 2: Clean up the file from the shared volume ---
                if (File.Exists(dataPathInContainer))
                {
                    File.Delete(dataPathInContainer);
                }
            }
        }
    }
}