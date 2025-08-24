using System;

namespace Temperance.Ludus.Models
{
    public class HistoricalPriceModel
    {
        public int SecurityID { get; set; }
        public string? Symbol { get; set; }
        public DateTime Timestamp { get; set; }
        public string? TimeInterval { get; set; }
        public decimal OpenPrice { get; set; }
        public decimal HighPrice { get; set; }
        public decimal LowPrice { get; set; }
        public decimal ClosePrice { get; set; }
        public decimal Volume { get; set; }
    }
}
