using RabbitMQ.Client;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Temperance.Ludus.Services.Interfaces;
using Tensorflow.Keras.Engine;

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
                HostName = configuration.GetValue<string>("RabbitMqSettings:HostName"),
                UserName = configuration.GetValue<string>("RabbitMqSettings:UserName"),
                Password = configuration.GetValue<string>("RabbitMqSettings:Password")
            };
            _connection = factory.CreateConnection();
            _channel = _connection.CreateModel();
            _logger.LogInformation("Successfully connected to RabbitMq");
        }

        public void Dispose()
        {
            _channel?.Dispose();
            _connection?.Dispose();
        }

        public void StartConsuming(string queueName, Func<string, Task> onMessageReceived)
        {
            _channel.QueueDeclare(Queue: queueName, durable: true, exclusive: false, autoDelete: false, arguments: null);

            var consumer = new EventingBasicCnsumer(_channel);
            consumer.Received += async (model, ea) =>
            {
                var body = ea.Body.ToArray();
                var message = Encoding.UTF8.GetString(body);
                _logger.LogInformation("Received message from queue: {QueueName}", queueName);

                try
                {
                    await onMessageReceived(message);
                    _channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
                    _logger.LogInformation("Processed message successfully: {Message}", message);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing message: {Message}", message);
                    _channel.BasicNack(deliveryTag: ea.DeliveryTag, multiple: false, requeue: true);
                }
            };

            _channel.BasicConsume(Queue: queueName, autoAck)zl
        }

        public void StartConsuming(string queueName, Func<string, Task> onMessageReceived)
        {
            throw new NotImplementedException();
        }
    }
}
