using System.Net.Http.Json;
using Temperance.Ludus.Models.Payloads;
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

        public async Task NotifyOptimizationCompleteAsync(Guid jobId, Guid sessionId)
        {
            _logger.LogInformation("Notifying Conductor of job completion for JobId: {JobId}", jobId);

            var payload = new OptimizationCompletePayload(jobId, sessionId);

            var response = await _httpClient.PostAsJsonAsync("api/Orchestration/notify-complete", payload);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                _logger.LogError("Failed to notify Conductor of job completion for JobId {JobId}. Status: {StatusCode}. Error: {Error}",
                    jobId, response.StatusCode, errorContent);
                response.EnsureSuccessStatusCode();
            }

            _logger.LogInformation("Successfully notified Conductor of job completion for JobId: {JobId}", jobId);
        }

        public async Task NotifyOptimizationFailedAsync(Guid jobId, Guid sessionId, string errorMessage)
        {
            _logger.LogWarning("Notifying Conductor of job failure for JobId: {JobId}", jobId);

            var payload = new OptimizationFailurePayload(jobId, sessionId, errorMessage);

            var response = await _httpClient.PostAsJsonAsync("api/Orchestration/notify-failed", payload);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                _logger.LogError("Failed to notify Conductor of job failure for JobId {JobId}. Status: {StatusCode}. Error: {Error}",
                    jobId, response.StatusCode, errorContent);
                response.EnsureSuccessStatusCode();
            }

            _logger.LogInformation("Successfully notified Conductor of job failure for JobId: {JobId}", jobId);
        }
    }
}
