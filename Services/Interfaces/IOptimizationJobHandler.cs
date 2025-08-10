using Temperance.Ludus.Models;

namespace Temperance.Ludus.Services.Interfaces
{
    public interface IOptimizationJobHandler
    {
        Task<OptimizationResult> ProcessJobAsync(OptimizationJob job);
    }
}
