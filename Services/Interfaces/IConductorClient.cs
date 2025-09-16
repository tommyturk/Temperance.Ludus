using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Temperance.Ludus.Services.Interfaces
{
    public interface IConductorClient
    {
        Task NotifyOptimizationCompleteAsync(Guid jobId, Guid sessionId);
        Task NotifyOptimizationFailedAsync(Guid jobId, Guid sessionId, string errorMessage);
    }
}
