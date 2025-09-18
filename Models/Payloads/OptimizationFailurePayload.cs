using System.Text.Json.Serialization;

namespace Temperance.Ludus.Models.Payloads
{
    public record OptimizationFailurePayload(
            Guid JobId,
            Guid SessionId,
            string Status,
            [property: JsonPropertyName("ErrorMessaage")] string ErrorMessage
        );
    public record OptimizationCompletePayload(
        Guid JobId,
        Guid SessionId,
        string Status,
        [property: JsonPropertyName("ErrorMessaage")] string ErrorMessage
    );
}
