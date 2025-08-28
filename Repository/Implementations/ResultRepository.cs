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

        public async Task<int> SaveOptimizationResultAsync(OptimizationResult result)
        {
            if (result.OptimizedParameters == null)
            {
                _logger.LogWarning("Cannot save result with null parameters for JobId {JobId}", result.JobId);
                return 0;
            }

            var connectionString = _configuration.GetConnectionString("DefaultConnection");
            // Corrected SQL: Removed the JobId column from the INSERT statement
            const string sql = @"
                INSERT INTO Ludus.StrategyOptimizedParameters 
                    (StrategyName, Symbol, Interval, OptimizedParametersJson)
                VALUES 
                    (@StrategyName, @Symbol, @Interval, @OptimizedParametersJson);
                SELECT CAST(SCOPE_IDENTITY() as int)";

            try
            {
                await using var connection = new SqlConnection(connectionString);
                var newId = await connection.ExecuteScalarAsync<int>(sql, new
                {
                    result.StrategyName,
                    result.Symbol,
                    result.Interval,
                    OptimizedParametersJson = JsonSerializer.Serialize(result.OptimizedParameters)
                });
                _logger.LogInformation("Successfully saved parameters for JobId {JobId} ({Strategy} on {Symbol}/{Interval}) with new record ID {Id}",
                    result.JobId, result.StrategyName, result.Symbol, result.Interval, newId);

                return newId;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save optimization result for JobId {JobId}", result.JobId);
                throw;
            }
        }
    }
}