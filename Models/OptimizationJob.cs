using System.Text.Json.Serialization;

namespace Temperance.Ludus.Models
{
    public class OptimizationJob
    {
        [JsonPropertyName("jobId")] // Add this attribute
        public Guid JobId { get; set; }

        [JsonPropertyName("sessionId")]
        public Guid? SessionId { get; set; }

        [JsonPropertyName("strategyName")]
        public string StrategyName { get; set; } = string.Empty;

        [JsonPropertyName("symbol")]
        public string Symbol { get; set; } = string.Empty;

        [JsonPropertyName("interval")]
        public string Interval { get; set; } = string.Empty;

        [JsonPropertyName("startDate")]
        public DateTime StartDate { get; set; }

        [JsonPropertyName("endDate")]
        public DateTime EndDate { get; set; }

        [JsonPropertyName("mode")]
        public int Mode { get; set; }

        [JsonPropertyName("epochs")]
        public int Epochs { get; set; } = 50;

        [JsonPropertyName("lookBack")]
        public int LookBack { get; set; } = 60;

        [JsonPropertyName("resultKey")]
        public string ResultKey { get; set; }
    }
}