using System;

namespace Temperance.Ludus.Models
{
    public class HistoricalPriceModel
    {
        public int SecurityID { get; set; }
        public string? Symbol { get; set; }
        public DateTime Timestamp { get; set; }
        public string? TimeInterval { get; set; }
        public double OpenPrice { get; set; }
        public double HighPrice { get; set; }
        public double LowPrice { get; set; }
        public double ClosePrice { get; set; }
        public long Volume { get; set; }
    }
}
