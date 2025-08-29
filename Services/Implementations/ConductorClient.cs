using System.Net.Http.Json;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class ConductorClient : IConductorClient
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<ConductorClient> _logger;

        public ConductorClient(HttpClient httpClient, IConfiguration configuration, ILogger<ConductorClient> logger)
        {
            _httpClient = httpClient;
            _logger = logger;
            _httpClient.BaseAddress = new Uri(configuration["ConductorApi:BaseUrl"]
                ?? "http://conductor:7200/"); 
        }

        public async Task TriggerBacktestAsync(OptimizationResult result)
        {
            _logger.LogInformation("Notifying Conductor to start backtest for JobId {JobId}", result.JobId);

            var payload = new
            {
                OptimizationId = result.Id,
                OptimizationJobId = result.JobId,
                SessionId = result.SessionId,
                StartDate = result.EndDate.AddDays(1),
                EndDate = result.EndDate.AddDays(7),
            };

            try
            {
                var response = await _httpClient.PostAsJsonAsync("api/backtest/start-from-optimization", payload);
                response.EnsureSuccessStatusCode();
                _logger.LogInformation("Successfully triggered backtest via Conductor for JobId {JobId}", result.JobId);
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "Failed to trigger backtest via Conductor for JobId {JobId}", result.JobId);
                // Depending on requirements, you might want to retry or handle this failure
                throw;
            }
        }
    }
}
