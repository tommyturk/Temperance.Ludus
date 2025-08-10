using static Temperance.Ludus.Services.Implementations.OptimizationJobHandler;

namespace Temperance.Ludus.Services.Interfaces
{
    public interface IOptimizationJobHandler
    {
        Task<OptimizationResult> ProcessJobAsync(OptimizationJob job);
    }
}
