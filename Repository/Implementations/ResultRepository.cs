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
                _logger.LogWarning("Cannot save result with null parameters for {Strategy}/{Symbol}",
                    result.StrategyName, result.Symbol);
                return 0;
            }

            var connectionString = _configuration.GetConnectionString("DefaultConnection");

            const string sql = @"
                INSERT INTO Ludus.StrategyOptimizedParameters 
                    (StrategyName, Symbol, Interval, OptimizedParametersJson, TotalReturns, StartDate, EndDate, CreatedAt)
                VALUES 
                    (@StrategyName, @Symbol, @Interval, @OptimizedParametersJson, @TotalReturns, @StartDate, @EndDate, @CreatedAt);
                SELECT CAST(SCOPE_IDENTITY() as int)";

            await using var connection = new SqlConnection(connectionString);

            var newId = await connection.ExecuteScalarAsync<int>(sql, new
            {
                result.StrategyName,
                result.Symbol,
                result.Interval,
                OptimizedParametersJson = JsonSerializer.Serialize(result.OptimizedParameters),
                result.TotalReturns,
                result.StartDate,
                result.EndDate,
                CreatedAt = DateTime.UtcNow 
            });

            _logger.LogInformation(
                "Successfully saved new universal parameters for {Strategy} on {Symbol}/{Interval} with new record ID {Id}",
                result.StrategyName, result.Symbol, result.Interval, newId);

            return newId;
        }
    }
}