using System.Globalization;
using System.Text;
using System.Text.Json;
using Temperance.Ludus.Models;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations;

public class OptimizationJobHandler : IOptimizationJobHandler
{
    private readonly ILogger<OptimizationJobHandler> _logger;
    private readonly IPythonScriptRunner _pythonScriptRunner;
    private readonly IHistoricalDataService _historicalDataService;

    public OptimizationJobHandler(
        ILogger<OptimizationJobHandler> logger,
        IPythonScriptRunner pythonScriptRunner,
        IHistoricalDataService historicalDataService)
    {
        _logger = logger;
        _pythonScriptRunner = pythonScriptRunner;
        _historicalDataService = historicalDataService;
    }

    public async Task ProcessJobAsync(OptimizationJob job)
    {
        _logger.LogInformation("Processing optimization job {JobId} for {Symbol} [{Interval}]...",
            job.JobId, job.Symbol, job.Interval);

        var inputCsvPath = Path.Combine(Path.GetTempPath(), $"{job.JobId}_input.csv");
        var outputJsonPath = Path.Combine(Path.GetTempPath(), $"{job.JobId}_output.json");

        try
        {
            var allPrices = await _historicalDataService.GetHistoricalPricesAsync(job.Symbol, job.Interval, job.StartDate, job.EndDate);

            var filteredPrices = allPrices
                .Where(p => p.Timestamp >= job.StartDate && p.Timestamp <= job.EndDate)
                .OrderBy(p => p.Timestamp)
                .ToList();

            if (!filteredPrices.Any())
            {
                _logger.LogWarning("No historical data found for job {JobId} within the specified date range. Skipping.", job.JobId);
                return;
            }

            var csvBuilder = new StringBuilder();
            csvBuilder.AppendLine("Timestamp,OpenPrice,HighPrice,LowPrice,ClosePrice,Volume");
            foreach (var price in filteredPrices)
            {
                csvBuilder.AppendLine(
                    $"{price.Timestamp:o}," + // ISO 8601 format
                    $"{price.OpenPrice.ToString(CultureInfo.InvariantCulture)}," +
                    $"{price.HighPrice.ToString(CultureInfo.InvariantCulture)}," +
                    $"{price.LowPrice.ToString(CultureInfo.InvariantCulture)}," +
                    $"{price.ClosePrice.ToString(CultureInfo.InvariantCulture)}," +
                    $"{price.Volume.ToString(CultureInfo.InvariantCulture)}"
                );
            }
            var historicalDataCsv = csvBuilder.ToString();

            await File.WriteAllTextAsync(inputCsvPath, historicalDataCsv);
            _logger.LogInformation("Wrote {Count} data points for job {JobId} to {InputPath}",
                filteredPrices.Count, job.JobId, inputCsvPath);

            var scriptArgs = new Dictionary<string, string>
            {
                { "input_csv_path", inputCsvPath },
                { "output_json_path", outputJsonPath }
            };
            await _pythonScriptRunner.RunScriptAsync("optimizer.py", scriptArgs);

            var resultJson = await File.ReadAllTextAsync(outputJsonPath);
            var result = JsonSerializer.Deserialize<PythonScriptResult>(resultJson,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            if (result is { Status: "success", BestParameters: not null, Performance: not null })
            {
                _logger.LogInformation(
                   "Successfully found optimal parameters for Job {JobId}. Performance: {Performance:F4}. Parameters: {Parameters}",
                   job.JobId, result.Performance, JsonSerializer.Serialize(result.BestParameters));

                var optimizationResult = new OptimizationResult
                {
                    Id = Guid.NewGuid(), // The primary key for this result entry
                    BacktestRunId = job.JobId,
                    StrategyName = "MeanReversion_BB_RSI", // You might pass this in the job later
                    Symbol = job.Symbol,
                    Interval = job.Interval,
                    ParametersJson = JsonSerializer.Serialize(result.BestParameters),
                    TotalReturn = (decimal)result.Performance, // Assuming performance is total return
                    // You would calculate other metrics like Sharpe Ratio, Win Rate etc. here if available
                    TimestampUtc = DateTime.UtcNow
                };

                await _resultRepository.SaveOptimizationResultAsync(optimizationResult);
                // --- END OF INTEGRATION ---
            }
            else
            {
                _logger.LogWarning("Optimization script for Job {JobId} reported an issue: {Message}",
                    job.JobId, result?.Message ?? "No message provided.");
            }
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