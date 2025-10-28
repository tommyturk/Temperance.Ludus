namespace Temperance.Ludus.Models
{
    public class OptimizationMetrics
    {
        public double? TotalReturns { get; set; }
        public double SharpeRatio { get; set; }
        public double? WinRate { get; set; }
        public int? NumTrades { get; set; }
        public double? MaxDrawdown { get; set; }
        public double? ProfitFactor { get; set; }
    }
}
