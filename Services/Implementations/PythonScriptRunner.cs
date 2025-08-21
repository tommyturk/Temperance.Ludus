using System.Collections;
using System.Diagnostics;
using System.Text;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class PythonScriptRunner : IPythonScriptRunner
    {
        private readonly ILogger<PythonScriptRunner> _logger;
        private const string ScriptBasePathInContainer = "/app/scripts/";

        public PythonScriptRunner(ILogger<PythonScriptRunner> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public async Task<(string Output, string Error)> RunScriptAsync(string scriptName, Dictionary<string, object> arguments)
        {
            var scriptPathInContainer = Path.Combine(ScriptBasePathInContainer, scriptName).Replace('\\', '/');

            var scriptArgsBuilder = new StringBuilder();
            foreach (var arg in arguments)
            {
                scriptArgsBuilder.Append($" --{arg.Key} ");

                if (arg.Value is IEnumerable enumerable && !(arg.Value is string))
                {
                    var values = new List<string>();
                    foreach (var item in enumerable)
                        values.Add(item.ToString());
                    scriptArgsBuilder.Append(string.Join(" ", values));
                }
                else
                    scriptArgsBuilder.Append($"\"{arg.Value}\"");
            }
            var processStartInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"{scriptPathInContainer}{scriptArgsBuilder}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = "/app"
            };

            _logger.LogInformation("Executing command: {FileName} {Arguments}", processStartInfo.FileName, processStartInfo.Arguments);

            using var process = new Process { StartInfo = processStartInfo };

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            process.OutputDataReceived += (sender, args) => { if (args.Data != null) outputBuilder.AppendLine(args.Data); };
            process.ErrorDataReceived += (sender, args) => { if (args.Data != null) errorBuilder.AppendLine(args.Data); };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            // Use a cancellation token for graceful shutdown if the host requests it
            await process.WaitForExitAsync();

            var output = outputBuilder.ToString();
            var error = errorBuilder.ToString();

            if (process.ExitCode != 0)
            {
                _logger.LogError("Python script {ScriptName} failed with exit code {ExitCode}. Error: {Error}", scriptName, process.ExitCode, error);
                throw new InvalidOperationException($"Python script {scriptName} failed. Error: {error}");
            }

            return (output, error);
        }
    }
}