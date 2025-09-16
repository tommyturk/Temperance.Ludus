namespace Temperance.Ludus.Models.Payloads
{
    public record OptimizationFailurePayload(Guid JobId, Guid SessionId, string ErrorMessage);
    public record OptimizationCompletePayload(Guid JobId, Guid SessionId);
}
