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
        private readonly string _sharedDataPath;

        // This record now matches the new JSON output from the Python script
        public record PythonOptimizationOutput(
            string Status,
            string Mode,
            Dictionary<string, object> OptimizedParameters,
            Dictionary<string, object> BruteForcePerformance
        );

        public OptimizationJobHandler(
            IHistoricalDataService historicalDataService,
            IPythonScriptRunner scriptRunner,
            IResultRepository resultRepository,
            ILogger<OptimizationJobHandler> logger,
            IConfiguration configuration,
            IHostEnvironment environment)
        {
            _historicalDataService = historicalDataService;
            _scriptRunner = scriptRunner;
            _resultRepository = resultRepository;
            _logger = logger;

            // This logic correctly handles local vs. Docker paths
            _sharedDataPath = "/temp_data";
            if (environment.IsDevelopment())
            {
                _sharedDataPath = configuration.GetValue<string>("PythonSettings:SharedDataPath")
                    ?? throw new InvalidOperationException("PythonSettings:SharedDataPath is not configured.");
            }
        }

        public async Task<OptimizationResult> ProcessJobAsync(OptimizationJob job)
        {
            // --- 1. Prepare Data ---
            _logger.LogInformation("Fetching historical data for job {JobId} ({Mode})", job.Id, job.Mode);
            var historicalData = await _historicalDataService.GetHistoricalPricesAsync(
                job.Symbol, job.Interval, job.StartDate, job.EndDate);

            if (historicalData == null || !historicalData.Any())
                return new OptimizationResult { JobId = job.Id, Status = "Failed: No Data" };

            var dataPath = Path.Combine(_sharedDataPath, $"{job.Id}.csv");
            var csvHeader = "Timestamp,OpenPrice,HighPrice,LowPrice,ClosePrice,Volume";
            var csvLines = historicalData.Select(p =>
                $"{p.Timestamp:O},{p.OpenPrice},{p.HighPrice},{p.LowPrice},{p.ClosePrice},{p.Volume}"
            );
            await File.WriteAllLinesAsync(dataPath, new[] { csvHeader }.Concat(csvLines));

            // --- 2. Define Script Arguments (THE FIX IS HERE) ---
            // This now includes all the arguments the new python script requires.
            var scriptArgs = new Dictionary<string, object>
            {
                { "data_path", dataPath },
                { "mode", job.Mode }, // <-- Required argument is now passed
                { "model_path", "/app/models/lstm_optimizer.h5" },
                { "epochs", job.Epochs },
                { "lookback", job.LookBack }
            };

            // Add the parameter ranges, these could come from the job or config in the future
            scriptArgs.Add("ma_period_range", new[] { 20, 201, 20 });
            scriptArgs.Add("rsi_period_range", new[] { 7, 31, 3 });
            scriptArgs.Add("rsi_oversold_range", new[] { 15, 41, 5 });
            scriptArgs.Add("rsi_overbought_range", new[] { 60, 86, 5 });
            scriptArgs.Add("std_dev_multiplier_range", new[] { 1.5, 3.6, 0.5 });
            scriptArgs.Add("atr_period_range", new[] { 7, 29, 3 });
            scriptArgs.Add("atr_multiplier_range", new[] { 1.0, 5.1, 0.5 });

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
                PerformanceMetrics = pythonResult.BruteForcePerformance, // Using brute-force performance for now
                Status = "Completed"
            };

            await _resultRepository.SaveOptimizationResultAsync(result);

            if (File.Exists(dataPath))
                File.Delete(dataPath);

            return result;
        }
    }
}