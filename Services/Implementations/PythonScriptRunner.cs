using System.Diagnostics;
using System.Text;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations;

public class PythonScriptRunner : IPythonScriptRunner
{
    private readonly ILogger<PythonScriptRunner> _logger;
    private readonly string _scriptsPath = Path.Combine(AppContext.BaseDirectory, "scripts");

    public PythonScriptRunner(ILogger<PythonScriptRunner> logger)
    {
        _logger = logger;
    }

    public async Task<string> RunScriptAsync(string scriptName, Dictionary<string, string> arguments)
    {
        var scriptPath = Path.Combine(_scriptsPath, scriptName);
        if (!File.Exists(scriptPath))
        {
            throw new FileNotFoundException($"Python script not found: {scriptPath}");
        }

        // Correctly format the arguments for the command line.
        // The script path is the first argument, followed by the dictionary values.
        var argsBuilder = new StringBuilder();
        argsBuilder.Append($"\"{scriptPath}\"");
        foreach (var arg in arguments)
        {
            if (arg.Key.StartsWith("-"))
                argsBuilder.Append($" {arg.Key} \"{arg.Value}\"");
            else
                argsBuilder.Append($" \"{arg.Value}\"");
        }

        var processStartInfo = new ProcessStartInfo
        {
            FileName = "python", // Assumes 'python' is in the system's PATH
            Arguments = argsBuilder.ToString(),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        _logger.LogInformation("Executing Python script: {FileName} {Arguments}",
            processStartInfo.FileName, processStartInfo.Arguments);

        using var process = new Process { StartInfo = processStartInfo };

        var stdOut = new StringBuilder();
        var stdErr = new StringBuilder();

        process.OutputDataReceived += (sender, args) => {
            if (args.Data != null) stdOut.AppendLine(args.Data);
        };
        process.ErrorDataReceived += (sender, args) => {
            if (args.Data != null) stdErr.AppendLine(args.Data);
        };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        await process.WaitForExitAsync();

        if (process.ExitCode != 0)
        {
            var errorMessage = stdErr.ToString();
            _logger.LogError("Python script {ScriptName} failed with exit code {ExitCode}. Error: {Error}",
                scriptName, process.ExitCode, errorMessage);
            // Throw the detailed error from Python to be handled upstream
            throw new InvalidOperationException($"Python script {scriptName} failed. Error: {errorMessage}");
        }

        _logger.LogInformation("Python script {ScriptName} executed successfully.", scriptName);
        return stdOut.ToString();
    }
}