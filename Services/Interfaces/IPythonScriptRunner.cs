namespace Temperance.Ludus.Services.Interfaces
{
    public interface IPythonScriptRunner
    {
        Task<string> RunScriptAsync(string scriptName, Dictionary<string, string> arguments);
    }
}

