using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Temperance.Ludus.Services.Interfaces
{
    public interface IConductorClient
    {
        Task TriggerBacktestAsync(OptimizationResult result);
    }
}
