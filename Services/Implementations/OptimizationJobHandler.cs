using System.Globalization;
using System.Text;
using System.Text.Json;
using Temperance.Ludus.Models;
using Temperance.Ludus.Repository.Interfaces;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class OptimizationJobHandler : IOptimizationJobHandler
    {
        private readonly ILogger<OptimizationJobHandler> _logger;
        private readonly IPythonScriptRunner _pythonScriptRunner;
        private readonly IHistoricalDataService _historicalDataService;
        private readonly IResultRepository _resultRepository;
        private readonly IConductorClient _conductorClient;

        public OptimizationJobHandler(
            ILogger<OptimizationJobHandler> logger,
            IPythonScriptRunner pythonScriptRunner,
            IHistoricalDataService historicalDataService,
            IResultRepository resultRepository,
            IConductorClient conductorClient)
        {
            _logger = logger;
            _pythonScriptRunner = pythonScriptRunner;
            _historicalDataService = historicalDataService;
            _resultRepository = resultRepository;
            _conductorClient = conductorClient;
        }

        public async Task<PythonScriptResult?> ProcessJobAsync(OptimizationJob job)
        {
            _logger.LogInformation("Processing optimization job {JobId} for {Symbol} [{Interval}]...",
                job.JobId, job.Symbol, job.Interval);

            var inputCsvPath = Path.Combine(Path.GetTempPath(), $"{job.JobId}_input.csv");
            var outputJsonPath = Path.Combine(Path.GetTempPath(), $"{job.JobId}_output.json");
            var modelDir = Path.Combine(Path.GetTempPath(), "ludus_models");
            Directory.CreateDirectory(modelDir);
            var modelPath = Path.Combine(modelDir, $"{job.StrategyName}_{job.Symbol}_{job.Interval}.keras".Replace("/", "_"));

            var jsonJob = JsonSerializer.Serialize(job, new JsonSerializerOptions { WriteIndented = true });
            _logger.LogInformation($"Job payload: {jsonJob}");
            try
            {
                var prices = await _historicalDataService.GetHistoricalPricesAsync(job.Symbol, job.Interval, job.StartDate, job.EndDate);
                if (!prices.Any())
                {
                    _logger.LogWarning("No historical data found for job {JobId} within the specified date range. Skipping.", job.JobId);
                    return new PythonScriptResult { Status = "Skipped", Message = "No historical data." };
                }

                var csvBuilder = new StringBuilder();
                csvBuilder.AppendLine("Timestamp,OpenPrice,HighPrice,LowPrice,ClosePrice,Volume");
                foreach (var price in prices)
                {
                    csvBuilder.AppendLine(
                        $"{price.Timestamp:o}," +
                        $"{price.OpenPrice.ToString(CultureInfo.InvariantCulture)}," +
                        $"{price.HighPrice.ToString(CultureInfo.InvariantCulture)}," +
                        $"{price.LowPrice.ToString(CultureInfo.InvariantCulture)}," +
                        $"{price.ClosePrice.ToString(CultureInfo.InvariantCulture)}," +
                        $"{price.Volume.ToString(CultureInfo.InvariantCulture)}"
                    );
                }
                await File.WriteAllTextAsync(inputCsvPath, csvBuilder.ToString());
                _logger.LogInformation("Wrote {Count} data points for job {JobId} to {InputPath}", prices.Count, job.JobId, inputCsvPath);

                var scriptArgs = new Dictionary<string, string>
                {
                    { "input_csv_path", inputCsvPath },
                    { "output_json_path", outputJsonPath },
                    { "--mode", job.Mode },
                    { "--model-path", modelPath },
                    { "--epochs", job.Epochs.ToString() },
                    { "--lookback", job.LookBack.ToString() }
                };

                await _pythonScriptRunner.RunScriptAsync("optimizer.py", scriptArgs);

                var resultJson = await File.ReadAllTextAsync(outputJsonPath);
                var result = JsonSerializer.Deserialize<PythonScriptResult>(resultJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (result is { Status: "success", BestParameters: not null })
                {
                    _logger.LogInformation(
                       "Successfully found optimal parameters for Job {JobId}. Performance: {Perf}. Parameters: {Params}",
                       job.JobId, result.Performance, JsonSerializer.Serialize(result.BestParameters));

                    var optimizationResult = new OptimizationResult
                    {
                        JobId = job.JobId,
                        SessionId = job.SessionId,
                        StrategyName = job.StrategyName,
                        Symbol = job.Symbol,
                        Interval = job.Interval,
                        OptimizedParameters = result.BestParameters,
                        StartDate = job.StartDate,
                        EndDate = job.EndDate
                    };

                    var optimizationRecordId = await _resultRepository.SaveOptimizationResultAsync(optimizationResult);
                    optimizationResult.Id = optimizationRecordId;
                    await _conductorClient.TriggerBacktestAsync(optimizationResult);
                }
                else
                {
                    _logger.LogWarning("Optimization script for Job {JobId} did not succeed: {Message}",
                        job.JobId, result?.Message ?? "No message provided.");
                }
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "An unhandled exception occurred while processing job {JobId}.", job.JobId);
                throw;
            }
            finally
            {
                if (File.Exists(inputCsvPath)) File.Delete(inputCsvPath);
                if (File.Exists(outputJsonPath)) File.Delete(outputJsonPath);
                _logger.LogDebug("Cleaned up temporary files for Job {JobId}", job.JobId);
            }
        }
    }
}