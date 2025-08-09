using System.Diagnostics;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class PythonScriptRunner : IPythonScriptRunner
    {
        private readonly ILogger<PythonScriptRunner> _logger;
        private readonly string _scriptPath;

        public PythonScriptRunner(ILogger<PythonScriptRunner> logger, IConfiguration configuration)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _scriptPath = configuration.GetValue<string>("PythonSettings:ScriptPath") ?? "/app/scripts/";
        }

        public async Task<string> RunScriptAsync(string scriptName, Dictionary<string, object> arguments)
        {
            var fullScriptPath = Path.Combine(_scriptPath, scriptName);
            var argsString = string.Join(" ", arguments.Select(kv => $"{kv.Key}={kv.Value}"));

            var startInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"{fullScriptPath} {argsString}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            _logger.LogInformation("Executing Python script: {FileName} {Arguments}", startInfo.FileName, startInfo.Arguments);

            using var process = Process.Start(startInfo);
            if(process == null)
                throw new InvalidOperationException("Failed to start Python process.");

            string output = await process.StandardOutput.ReadToEndAsync();
            string error = await process.StandardError.ReadToEndAsync();    

            await process.WaitForExitAsync();

            if(process.ExitCode != 0)
            {
                _logger.LogError("Python script {ScriptName} failed with exit code {ExitCode}. Error: {Error}", scriptName, process.ExitCode, error);
                throw new InvalidOperationException($"Python script {scriptName} failed with exit code {process.ExitCode}. Error: {error}");
            }

            _logger.LogInformation("Python script {ScriptName} completed successfully. Output: {Output}", scriptName, output);
            return output;
        }
    }
}
