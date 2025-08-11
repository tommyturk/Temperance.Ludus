using Dapper;
using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System;
using System.Text.Json;
using System.Threading.Tasks;
using Temperance.Ludus.Models;
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
            if (result.Status != "Completed" || result.OptimizedParameters == null)
            {
                _logger.LogWarning("Cannot save non-completed or empty result for JobId {JobId}", result.JobId);
                return;
            }

            var connectionString = _configuration.GetConnectionString("HistoricalDatabase");

            const string sql = @"
                MERGE dbo.StrategyOptimizedParameters AS target
                USING (VALUES (@StrategyName, @Symbol, @Interval, @OptimizedParametersJson)) 
                    AS source (StrategyName, Symbol, Interval, OptimizedParametersJson)
                ON target.StrategyName = source.StrategyName AND target.Symbol = source.Symbol AND target.Interval = source.Interval
                WHEN MATCHED THEN
                    UPDATE SET 
                        OptimizedParametersJson = source.OptimizedParametersJson,
                        CreatedAt = GETUTCDATE()
                WHEN NOT MATCHED THEN
                    INSERT (StrategyName, Symbol, Interval, OptimizedParametersJson)
                    VALUES (source.StrategyName, source.Symbol, source.Interval, source.OptimizedParametersJson);
            ";

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
                _logger.LogInformation("Successfully saved optimized parameters for {Strategy} on {Symbol}/{Interval}", result.StrategyName, result.Symbol, result.Interval);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save optimization result for JobId {JobId}", result.JobId);
                throw;
            }
        }
    }
}