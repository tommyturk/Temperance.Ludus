using Temperance.Ludus.Models;

namespace Temperance.Ludus.Services.Interfaces
{
    public interface IHistoricalDataService
    {
        Task<List<HistoricalPriceModel>> GetHistoricalPricesAsync(string symbol, string interval, DateTime start, DateTime end);
    }
}
