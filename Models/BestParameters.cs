namespace Temperance.Ludus.Models
{
    public class BestParameters
    {
        public int MovingAveragePeriod { get; set; }
        public double StdDevMultiplier { get; set; }
        public int RSIPeriod { get; set; }
        public int RSIOverbought { get; set; }
        public int RSIOversold { get; set; }
    }
}
