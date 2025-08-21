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

            // A temporary file is still created on the HOST machine's temp directory
            var tempFilePathOnHost = Path.Combine(Path.GetTempPath(), $"{job.Id}.csv");

            try
            {
                // --- CHANGE 1: Correct CSV Header for backtesting.py ---
                // The backtesting library expects these specific column names.
                var csvHeader = "Timestamp,Open,High,Low,Close,Volume";
                var csvLines = historicalData.Select(p =>
                    $"{p.Timestamp:O},{p.OpenPrice},{p.HighPrice},{p.LowPrice},{p.ClosePrice},{p.Volume}"
                );
                await File.WriteAllLinesAsync(tempFilePathOnHost, new[] { csvHeader }.Concat(csvLines));

                // --- CHANGE 2: Host-to-Container Path Translation ---
                // Get just the filename from the full host path.
                var fileNameOnly = Path.GetFileName(tempFilePathOnHost);
                // Construct the path as it will appear INSIDE the container via the volume mount.
                var dataPathInContainer = $"/temp_data/{fileNameOnly}";

                // --- CHANGE 3: Streamlined Script Arguments ---
                // These now match the argparse arguments in the new optimizer.py script.
                // The dictionary should be <string, string> to match the new runner.
                var scriptArgs = new Dictionary<string, object>
                {
                    { "symbol", job.Symbol },
                    { "interval", job.Interval },
                    { "data_path", dataPathInContainer } // Pass the container-aware path
                };

                var (pythonOutput, pythonError) = await _scriptRunner.RunScriptAsync("optimizer.py", scriptArgs);

                // --- CHANGE 4: Robust JSON Deserialization ---
                // Find the JSON line in the script's output and parse it into our strongly-typed object.
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

                // Construct the final result from the parsed Python output
                var result = new OptimizationResult
                {
                    JobId = job.Id,
                    StrategyName = job.StrategyName,
                    Symbol = job.Symbol,
                    Interval = job.Interval,
                    OptimizedParameters = pythonResult.OptimizedParameters,
                    PerformanceMetrics = pythonResult.Performance, // Assuming you have a place to store this
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
                // This is crucial: always clean up the temporary file on the HOST
                if (File.Exists(tempFilePathOnHost))
                {
                    File.Delete(tempFilePathOnHost);
                }
            }
        }
    }
}
