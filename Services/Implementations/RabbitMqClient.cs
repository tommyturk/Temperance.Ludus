using RabbitMQ.Client;
using RabbitMQ.Client.Events;
using System.Text;
using Temperance.Ludus.Services.Interfaces;

namespace Temperance.Ludus.Services.Implementations
{
    public class RabbitMqClient : IMessageBusClient, IDisposable
    {
        private readonly ILogger<RabbitMqClient> _logger;
        private readonly IConnection _connection;
        private readonly IModel _channel;

        public RabbitMqClient(IConfiguration configuration, ILogger<RabbitMqClient> logger)
        {
            _logger = logger;
            var factory = new ConnectionFactory
            {
                // CORRECTED: Using "RabbitMQ" to match docker-compose.yml
                HostName = configuration.GetValue<string>("RabbitMQ:HostName"),
                UserName = configuration.GetValue<string>("RabbitMQ:UserName"),
                Password = configuration.GetValue<string>("RabbitMQ:Password")
            };

            _connection = factory.CreateConnection();
            _channel = _connection.CreateModel();
            _logger.LogInformation("Successfully connected to RabbitMQ.");
        }

        public void StartConsuming(string queueName, Func<string, Task> onMessageReceived)
        {
            _channel.QueueDeclare(queue: queueName, durable: true, exclusive: false, autoDelete: false, arguments: null);

            var consumer = new EventingBasicConsumer(_channel);
            consumer.Received += async (model, ea) =>
            {
                var body = ea.Body.ToArray();
                var message = Encoding.UTF8.GetString(body);
                _logger.LogInformation("Received message from queue '{queueName}'", queueName);

                try
                {
                    await onMessageReceived(message);
                    _channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing message. Re-queueing.");
                    _channel.BasicNack(deliveryTag: ea.DeliveryTag, multiple: false, requeue: true);
                }
            };

            // CORRECTED: Full, valid call to BasicConsume
            _channel.BasicConsume(queue: queueName, autoAck: false, consumer: consumer);
            _logger.LogInformation("Consumer started. Listening to queue '{queueName}'...", queueName);
        }

        public void Dispose()
        {
            _channel?.Dispose();
            _connection?.Dispose();
        }
    }
}
