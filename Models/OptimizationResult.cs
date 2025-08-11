using System;
using System.Collections.Generic;

namespace Temperance.Ludus.Models
{
    public class OptimizationResult
    {
        public Guid JobId { get; set; }
        public string StrategyName { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string Interval { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public Dictionary<string, object>? OptimizedParameters { get; set; }
    }
}
