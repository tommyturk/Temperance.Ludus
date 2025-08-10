using Temperance.Ludus.Models;

namespace Temperance.Ludus.Repository.Interfaces
{
    public interface IResultRepository
    {
        Task SaveOptimizationResultAsync(OptimizationResult result);
    }
}
