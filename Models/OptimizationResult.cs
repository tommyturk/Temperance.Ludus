public class OptimizationResult
{
    public Guid Id { get; set; }
    public Guid BacktestRunId { get; set; }
    public string StrategyName { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Interval { get; set; } = string.Empty;
    public string ParametersJson { get; set; } = string.Empty;
    public decimal TotalReturn { get; set; }
    public DateTime TimestampUtc { get; set; }
}