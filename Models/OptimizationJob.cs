using System;

namespace Temperance.Ludus.Models
{
    public class OptimizationJob
    {
        public Guid Id { get; set; }
        public string StrategyName { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string Interval { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public string Mode { get; set; } = "finetune";
        public int Epochs { get; set; } = 50;
        public int LookBack { get; set; } = 60;
    }
}
