using static Temperance.Ludus.Services.Implementations.OptimizationJobHandler;

namespace Temperance.Ludus.Repository.Interfaces
{
    public interface IResultRepository
    {
        Task SaveOptimizationResultAsync(OptimizationResult result);
    }
}
