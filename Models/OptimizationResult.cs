using Temperance.Ludus.Models;

public class OptimizationResult
{
    public Guid JobId { get; set; }
    public int Id { get; set; }
    public string StrategyName { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Interval { get; set; } = string.Empty;
    public BestParameters? OptimizedParameters { get; set; }
}