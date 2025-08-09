using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Temperance.Ludus.Services.Interfaces
{
    public interface IPythonScriptRunner
    {
        Task<string> RunScriptAsync(string scriptName, Dictionary<string, object> arguments);
    }
}
