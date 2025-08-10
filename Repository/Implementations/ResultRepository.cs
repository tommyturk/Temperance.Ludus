using Temperance.Ludus.Repository.Interfaces;
using Temperance.Ludus.Services.Implementations;

namespace Temperance.Ludus.Repository.Implementations
{
    public class ResultRepository : IResultRepository
    {
        private readonly ILogger<ResultRepository> _logger; 

        public ResultRepository(ILogger<ResultRepository> logger)
        {
            _logger = logger;
        }

        public Task SaveOptimizationResultAsync(OptimizationJobHandler.OptimizationResult result)
        {
            throw new NotImplementedException();
        }
    }
}
