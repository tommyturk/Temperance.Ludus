namespace Temperance.Ludus.Models
{
    public class PythonScriptResult
    {
        public string Status { get; set; }
        public BestParameters? BestParameters { get; set; }
        public OptimizationMetrics Metrics { get; set; }
        public string? Message { get; set; }
    }
}
