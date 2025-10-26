using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Options;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Temperance.Ludus.Confguration;
using Temperance.Ludus.Models;
using Temperance.Ludus.Repository.Interfaces;
using Temperance.Ludus.Services.Interfaces;
using Temperance.Ludus.Settings;

namespace Temperance.Ludus.Services.Implementations
{
    public class OptimizationJobHandler : IOptimizationJobHandler
    {
        private readonly ILogger<OptimizationJobHandler> _logger;
        private readonly IPythonScriptRunner _pythonScriptRunner;
        private readonly IHistoricalDataService _historicalDataService;
        private readonly IResultRepository _resultRepository;
        private readonly IConductorClient _conductorClient;
        private readonly PythonRunnerSettings _pythonRunnerSettings;
        private readonly OptimizerSettings _optimizerSettings;

        private readonly string _baseTempPath;

        public OptimizationJobHandler(
            ILogger<OptimizationJobHandler> logger,
            IPythonScriptRunner pythonScriptRunner,
            IHistoricalDataService historicalDataService,
            IResultRepository resultRepository,
            IConductorClient conductorClient,
            IOptions<PythonRunnerSettings> pythonRunnerSettings,
            IOptions<FilePathsSettings> filePathSettings,
            IOptions<OptimizerSettings> optimizerSettings)
        {
            _logger = logger;
            _pythonRunnerSettings = pythonRunnerSettings.Value;
            _historicalDataService = historicalDataService;
            _resultRepository = resultRepository;
            _conductorClient = conductorClient;
            _pythonScriptRunner = pythonScriptRunner;
            _baseTempPath = filePathSettings.Value.TempData;
            _optimizerSettings = optimizerSettings.Value;
        }

        public async Task<PythonScriptResult?> ProcessJobAsync(OptimizationJob job)
        {
            _logger.LogInformation("Processing optimization job {JobId} for {Symbol} [{Interval}]...",
                job.JobId, job.Symbol, job.Interval);

            _logger.LogInformation("Using strategy: {StrategyName}", job.StrategyName);
            var loadedKeys = string.Join(", ", _optimizerSettings.StrategyScripts.Keys);
            _logger.LogInformation("Available script keys in configuration: [{LoadedKeys}]", loadedKeys);

            //if (!_optimizerSettings.StrategyScripts.TryGetValue(job.StrategyName, out var scriptName))
            //{
            //    _logger.LogError("No optimizer script configured for strategy '{StrategyName}'. Cannot process job {JobId}.",
            //        job.StrategyName, job.JobId);

            //    await _conductorClient.NotifyOptimizationFailedAsync(job.JobId, job.SessionId.Value,
            //        $"No optimizer script configured for strategy: {job.StrategyName}");
            //}
            //if(String.IsNullOrWhiteSpace(scriptName))
            var scriptName = "optimizer_2.py";
            _logger.LogInformation("Using optimization script: {ScriptName} for strategy: {StrategyName}",
                scriptName, job.StrategyName);
            const int minimumDataPoints = 150; 

            var baseTempPath = _baseTempPath;
            Directory.CreateDirectory(baseTempPath);

            var inputCsvPath = Path.Combine(baseTempPath, $"{job.JobId}_input.csv");
            var outputJsonPath = Path.Combine(baseTempPath, $"{job.JobId}_output.json");

            var modelDir = Path.Combine(baseTempPath, "ludus_models");
            Directory.CreateDirectory(modelDir);

            var modelPath = Path.Combine(modelDir, $"{job.StrategyName}_{job.Symbol}_{job.Interval}.keras".Replace("/", "_"));

            var jsonJob = JsonSerializer.Serialize(job, new JsonSerializerOptions { WriteIndented = true });
            _logger.LogInformation($"Job payload: {jsonJob}");
            try
            {
                var prices = await _historicalDataService.GetHistoricalPricesAsync(job.Symbol, job.Interval, job.StartDate, job.EndDate);
                if (!prices.Any() || prices.Count < minimumDataPoints)
                {
                    _logger.LogWarning("No historical data found for job {JobId} within the specified date range. Skipping.", job.JobId);
                   
                    await _conductorClient.NotifyOptimizationFailedAsync(job.JobId, job.SessionId.Value,
                        $"No historical data found for job: Symbol: {job.Symbol} at Interval: {job.Interval}");

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

                var modeAsString = job.Mode == 0 ? "train" : "fine-tune";
                var scriptArgs = new Dictionary<string, string>
                {
                    { "input_csv_path", inputCsvPath },
                    { "output_json_path", outputJsonPath },
                    { "--mode", modeAsString },
                    { "--model-path", modelPath },
                    { "--epochs", job.Epochs.ToString() },
                    { "--lookback", job.LookBack.ToString() }
                };

                await _pythonScriptRunner.RunScriptAsync(scriptName, scriptArgs);

                var resultJson = await File.ReadAllTextAsync(outputJsonPath);
                var result = JsonSerializer.Deserialize<PythonScriptResult>(resultJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (result is { Status: "success", BestParameters: not null })
                {
                    var optimizationResult = new OptimizationResult
                    {
                        StrategyName = job.StrategyName,
                        Symbol = job.Symbol,
                        Interval = job.Interval,
                        OptimizedParameters = result.BestParameters,
                        StartDate = job.StartDate,
                        EndDate = job.EndDate
                    };

                    try
                    {
                        await _resultRepository.SaveOptimizationResultAsync(optimizationResult);
                    }
                    catch (SqlException ex) when (ex.Number == 2601 || ex.Number == 2627) 
                    {
                        _logger.LogWarning("Optimization result for {Symbol} already exists. Discarding redundant result.", job.Symbol);
                    }

                    await _conductorClient.NotifyOptimizationCompleteAsync(job.JobId, job.SessionId.Value);
                }
                else
                {
                    _logger.LogWarning("Optimization script for Job {JobId} did not succeed: {Message}",
                        job.JobId, result?.Message ?? "No message provided.");
                    await _conductorClient.NotifyOptimizationFailedAsync(job.JobId, job.SessionId.Value, "Optimization script failed.");
                }
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "An unhandled exception occurred while processing job {JobId}.", job.JobId);

                await _conductorClient.NotifyOptimizationFailedAsync(job.JobId, job.SessionId.Value, 
                    $"An unhandled exception occurred in Ludus: {ex.Message}");
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