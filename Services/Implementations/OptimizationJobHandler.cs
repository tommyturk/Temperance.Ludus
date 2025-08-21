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
            // --- 1. Prepare Data ---
            _logger.LogInformation("Fetching historical data for job {JobId} ({Mode})", job.Id, job.Mode);
            var historicalData = await _historicalDataService.GetHistoricalPricesAsync(
                job.Symbol, job.Interval, job.StartDate, job.EndDate);

            if (historicalData == null || !historicalData.Any())
                return new OptimizationResult { JobId = job.Id, Status = "Failed: No Data" };

            var dataPathInContainer = Path.Combine(SharedDataPathInContainer, $"{job.Id}.csv");
            var csvHeader = "Timestamp,OpenPrice,HighPrice,LowPrice,ClosePrice,Volume";
            var csvLines = historicalData.Select(p =>
                $"{p.Timestamp:O},{p.OpenPrice},{p.HighPrice},{p.LowPrice},{p.ClosePrice},{p.Volume}"
            );
            await File.WriteAllLinesAsync(dataPathInContainer, new[] { csvHeader }.Concat(csvLines));

            // --- 2. Define Parameter Ranges based on Mode ---
            var scriptArgs = new Dictionary<string, object>
            {
                { "data_path", dataPathInContainer }
            };

            if (job.Mode.Equals("train", StringComparison.OrdinalIgnoreCase))
            {
                _logger.LogInformation("Configuring WIDE parameter search for 'Train' mode.");
                scriptArgs.Add("ma_period_range", new[] { 20, 201, 20 });
                scriptArgs.Add("rsi_period_range", new[] { 7, 31, 3 });
                scriptArgs.Add("rsi_oversold_range", new[] { 15, 41, 5 });
                scriptArgs.Add("rsi_overbought_range", new[] { 60, 86, 5 });
                scriptArgs.Add("std_dev_multiplier_range", new[] { 1.5, 3.6, 0.5 });
                scriptArgs.Add("atr_period_range", new[] { 7, 29, 3 });
                scriptArgs.Add("atr_multiplier_range", new[] { 1.0, 5.1, 0.5 });
            }
            else if (job.Mode.Equals("finetune", StringComparison.OrdinalIgnoreCase))
            {
                _logger.LogInformation("Configuring FOCUSED parameter search for 'Finetune' mode.");
                scriptArgs.Add("ma_period_range", new[] { 40, 81, 10 });
                scriptArgs.Add("rsi_period_range", new[] { 10, 25, 2 });
                scriptArgs.Add("rsi_oversold_range", new[] { 25, 36, 2 });
                scriptArgs.Add("rsi_overbought_range", new[] { 65, 76, 2 });
                scriptArgs.Add("std_dev_multiplier_range", new[] { 1.8, 2.9, 0.2 });
                scriptArgs.Add("atr_period_range", new[] { 10, 22, 2 });
                scriptArgs.Add("atr_multiplier_range", new[] { 2.0, 4.1, 0.25 });
            }
            else
            {
                return new OptimizationResult { JobId = job.Id, Status = $"Failed: Unknown mode '{job.Mode}'" };
            }

            // --- 3. Execute the GPU Optimizer ---
            _logger.LogInformation("Invoking GPU optimizer script 'optimizer.py'...");
            var (pythonOutput, pythonError) = await _scriptRunner.RunScriptAsync("optimizer.py", scriptArgs);

            // --- 4. Process Results ---
            if (!string.IsNullOrWhiteSpace(pythonError))
            {
                _logger.LogError("Python script failed. Error: {Error}", pythonError);
                return new OptimizationResult { JobId = job.Id, Status = $"Failed: {pythonError}" };
            }

            var jsonLine = pythonOutput.Split('\n').FirstOrDefault(line => line.Trim().StartsWith("{") && line.Trim().EndsWith("}"));
            if (string.IsNullOrWhiteSpace(jsonLine))
            {
                throw new InvalidOperationException($"Python script did not return valid JSON. Output: {pythonOutput}");
            }

            var pythonResult = JsonSerializer.Deserialize<PythonOptimizationOutput>(jsonLine, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

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

            if (File.Exists(dataPathInContainer))
                File.Delete(dataPathInContainer);

            return result;
        }
    }
}