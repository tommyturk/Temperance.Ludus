using Microsoft.Extensions.Options;
using System.Diagnostics;
using System.Text;
using Temperance.Ludus.Confguration;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class PythonScriptRunner : IPythonScriptRunner
    {
        private readonly ILogger<PythonScriptRunner> _logger;
        private readonly string _ludusContainerName;
        private const string SCriptBasePathInContainer = "/app/scripts/";

        public PythonScriptRunner(ILogger<PythonScriptRunner> logger, IOptions<PythonRunnerSettings> settings)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _ludusContainerName = settings?.Value?.ContainerName
                ?? throw new ArgumentNullException(nameof(settings.Value.ContainerName));
        }

        public async Task<(string Output, string Error)> RunScriptAsync(string scriptName, Dictionary<string, object> arguments)
        {
            var scriptPathInContainer = Path.Combine(SCriptBasePathInContainer, scriptName).Replace('\\', '/');
            var scriptArgsBuilder = new StringBuilder();
            foreach(var arg in arguments)
                scriptArgsBuilder.Append($"--{arg.Key} \"{arg.Value}\" ");
            
            var commandToExecute = $"python {scriptPathInContainer}{scriptArgsBuilder}";

            var processStartInfo = new ProcessStartInfo
            {
                FileName = "docker", // We are now executing the 'docker' command
                                     // Arguments: exec [container_name] [command_to_run_inside]
                Arguments = $"exec {_ludusContainerName} {commandToExecute}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            _logger.LogInformation("Executing Docker command: {FileName} {Arguments}", processStartInfo.FileName, processStartInfo.Arguments);

            using var process = new Process { StartInfo = processStartInfo };

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            process.OutputDataReceived += (sender, args) => { if (args.Data != null) outputBuilder.AppendLine(args.Data); };
            process.ErrorDataReceived += (sender, args) => { if (args.Data != null) errorBuilder.AppendLine(args.Data); };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            await process.WaitForExitAsync();

            var output = outputBuilder.ToString();
            var error = errorBuilder.ToString();

            if (process.ExitCode != 0)
            {
                _logger.LogError("Docker command for script {ScriptName} failed with exit code {ExitCode}. Error: {Error}", scriptName, process.ExitCode, error);
                throw new InvalidOperationException($"Docker command for script {scriptName} failed with exit code {process.ExitCode}. Error: {error}");
            }

            return (output, error);
        }
    }
}
