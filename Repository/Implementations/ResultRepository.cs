using Dapper;
using Microsoft.Data.SqlClient;
using System.Text.Json;
using Temperance.Ludus.Repository.Interfaces;

namespace Temperance.Ludus.Repository.Implementations
{
    public class ResultRepository : IResultRepository
    {
        private readonly IConfiguration _configuration;
        private readonly ILogger<ResultRepository> _logger;

        public ResultRepository(IConfiguration configuration, ILogger<ResultRepository> logger)
        {
            _configuration = configuration;
            _logger = logger;
        }

        public async Task SaveOptimizationResultAsync(OptimizationResult result)
        {
            if (result.OptimizedParameters == null)
            {
                _logger.LogWarning("Cannot save result with null parameters for JobId {JobId}", result.JobId);
                return;
            }

            var connectionString = _configuration.GetConnectionString("DefaultConnection");
            const string sql = @"
                MERGE Ludus.StrategyOptimizedParameters AS target
                USING (SELECT @StrategyName AS StrategyName, @Symbol AS Symbol, @Interval AS Interval) AS source
                ON target.StrategyName = source.StrategyName AND target.Symbol = source.Symbol AND target.Interval = source.Interval
                WHEN MATCHED THEN
                    UPDATE SET OptimizedParametersJson = @OptimizedParametersJson, CreatedAt = GETUTCDATE()
                WHEN NOT MATCHED THEN
                    INSERT (StrategyName, Symbol, Interval, OptimizedParametersJson)
                    VALUES (@StrategyName, @Symbol, @Interval, @OptimizedParametersJson);";

            try
            {
                await using var connection = new SqlConnection(connectionString);
                await connection.ExecuteAsync(sql, new
                {
                    result.StrategyName,
                    result.Symbol,
                    result.Interval,
                    OptimizedParametersJson = JsonSerializer.Serialize(result.OptimizedParameters)
                });
                _logger.LogInformation("Successfully saved/updated parameters for {Strategy} on {Symbol}/{Interval}",
                    result.StrategyName, result.Symbol, result.Interval);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save optimization result for JobId {JobId}", result.JobId);
                throw;
            }
        }
    }
}