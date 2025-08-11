using Dapper;
using Microsoft.Data.SqlClient;
using Temperance.Ludus.Models;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class HistoricalDataService : IHistoricalDataService
    {
        private readonly IConfiguration _configuration;
        private readonly ILogger<HistoricalDataService> _logger;
        public HistoricalDataService(IConfiguration configuration, ILogger<HistoricalDataService> logger)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _logger = logger;
        }

        public async Task<List<HistoricalPriceModel>> GetHistoricalPricesAsync(string symbol, string interval, DateTime start, DateTime end)
        {
            var connectionString = _configuration.GetConnectionString("HistoricalDatabase");
            var tableName = $"Prices.{symbol}_{interval}";

            if (!tableName.All(c => char.IsLetterOrDigit(c) || c == '_' || c == '.'))
            {
                _logger.LogError("Invalid symbol or interval resulting in unsafe table name: {tableName}", tableName);
                throw new ArgumentException("Invalid symbol or interval provided.");
            }

            var sql = $"SELECT * FROM {tableName} WHERE Timestamp >= @StartDate AND Timestamp <= @EndDate ORDER BY Timestamp ASC";

            try
            {
                await using var connection = new SqlConnection(connectionString);
                var prices = await connection.QueryAsync<HistoricalPriceModel>(sql, new { StartDate = start, EndDate = end });
                return prices.AsList();
            }
            catch (SqlException ex)
            {
                _logger.LogError(ex, "Failed to fetch historical data for {symbol} [{interval}] from table {tableName}.", symbol, interval, tableName);
                return new List<HistoricalPriceModel>();
            }
        }
    }
}

