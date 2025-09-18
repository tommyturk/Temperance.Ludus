using System.Threading.Tasks;
using Temperance.Ludus.Models;

namespace Temperance.Ludus.Repository.Interfaces
{
    public interface IResultRepository
    {
        Task<int> SaveOptimizationResultAsync(OptimizationResult result, string resultKey);
    }
}
