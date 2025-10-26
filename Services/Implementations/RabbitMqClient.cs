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
        private readonly string _hostName;
        private readonly string _userName;
        private readonly string _password;

        public RabbitMqClient(IConfiguration configuration, ILogger<RabbitMqClient> logger)
        {
            _logger = logger;
            _hostName = configuration?.GetValue<string>("RabbitMQ:HostName") ?? "localhost";
            _userName = configuration?.GetValue<string>("RabbitMQ:UserName") ?? "guest";
            _password = configuration?.GetValue<string>("RabbitMQ:Password") ?? "guest";

            _logger.LogInformation("Attempting to connect to RabbitMQ at {HostName} with user {UserName}", _hostName, _userName);

            var factory = new ConnectionFactory
            {
                HostName = _hostName,
                UserName = _userName,
                Password = _password,
                AutomaticRecoveryEnabled = true,
                NetworkRecoveryInterval = TimeSpan.FromSeconds(10)
            };

            try
            {
                _connection = factory.CreateConnection();
                _channel = _connection.CreateModel();
                _logger.LogInformation("Successfully connected to RabbitMQ at {HostName}", _hostName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to connect to RabbitMQ at {HostName}", _hostName);
                throw;
            }
        }

        public void StartConsuming(string queueName, Func<string, Task> onMessageReceived, ushort prefetchCount = 16)
        {
            try
            {
                _logger.LogInformation("Declaring queue '{QueueName}'", queueName);
                var deadLetterExchangeName = "optimization_jobs_dlx";
                var deadLetterQueueName = "optimization_jobs_dlq";

                _channel.ExchangeDeclare(deadLetterExchangeName, ExchangeType.Direct);
                _channel.QueueDeclare(queue: deadLetterQueueName, durable: true, exclusive: false, autoDelete: false);
                _channel.QueueBind(deadLetterQueueName, deadLetterExchangeName, routingKey: queueName);

                var arguments = new Dictionary<string, object>()
                {
                    { "x-dead-letter-exchange", deadLetterExchangeName },
                    { "x-dead-letter-routing-key", queueName }
                };

                _channel.QueueDeclare(queue: queueName, durable: true, exclusive: false, autoDelete: false, arguments: null);

                _channel.BasicQos(prefetchSize: 0, prefetchCount: 16, global: false);
                _logger.LogInformation("Consumer QoS set with prefetch count of {PrefetchCount}", prefetchCount);

                var consumer = new EventingBasicConsumer(_channel);
                consumer.Received += async (model, ea) =>
                {
                    var body = ea.Body.ToArray();
                    var message = Encoding.UTF8.GetString(body);
                    _logger.LogInformation("Received message from queue '{QueueName}': {MessageLength} characters", queueName, message.Length);

                    try
                    {
                        await onMessageReceived(message);

                        _channel.BasicAck(deliveryTag: ea.DeliveryTag, multiple: false);
                        _logger.LogDebug("Message acknowledged successfully");
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing message from queue '{QueueName}'. Discarding message (requeue=false).", queueName);

                        _channel.BasicNack(deliveryTag: ea.DeliveryTag, multiple: false, requeue: false);
                    }
                };

                _channel.BasicConsume(queue: queueName, autoAck: false, consumer: consumer);
                _logger.LogInformation("Consumer started successfully. Listening to queue '{QueueName}'...", queueName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start consuming from queue '{QueueName}'", queueName);
                throw;
            }
        }

        public void Dispose()
        {
            try
            {
                _logger.LogInformation("Disposing RabbitMQ connection...");
                _channel?.Close();
                _channel?.Dispose();
                _connection?.Close();
                _connection?.Dispose();
                _logger.LogInformation("RabbitMQ connection disposed successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing RabbitMQ connection");
            }
        }
    }
}
