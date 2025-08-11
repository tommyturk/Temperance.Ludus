using System;
using System.Threading.Tasks;

namespace Temperance.Ludus.Services.Interfaces
{
    public interface IMessageBusClient
    {
        void StartConsuming(string queueName, Func<string, Task> onMessageReceived);
    }
}
