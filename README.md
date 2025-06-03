# Temperance.Ludus

Ludus: Conditional Parameter Optimization Worker Service
Project Overview
Ludus is a dedicated .NET 9.0 Worker Service within the Temperance algorithmic trading ecosystem, designed for the rigorous Conditional Parameter Optimization (CPO) of trading strategies. Its primary mission is to systematically refine and optimize strategy parameters based on prevailing market conditions and historical data, transforming raw market information into actionable, optimized configurations for live trading.

Inspired by the ancient Roman gladiator training schools, Ludus embodies the disciplined process of honing and perfecting performance. Just as a gladiator is trained to achieve peak effectiveness, Ludus trains machine learning models to extract optimal trading parameters from market chaos, ensuring our strategies are always operating with their sharpest edge.

Core Responsibilities:
Machine Learning Model Training: Develops, trains, and validates machine learning models (primarily leveraging TensorFlow.NET for GPU acceleration) to identify optimal parameters for various trading strategies.
Conditional Parameter Optimization (CPO): Executes sophisticated optimization routines that analyze historical market data to predict the best strategy parameters for future trading periods (e.g., for the next market open).
Asynchronous Processing: Operates as a background service, handling computationally intensive tasks without blocking core trading operations.
Scalable Compute: Designed for scalability, capable of leveraging multiple CPU cores and, crucially, NVIDIA GPUs (on local machines and future Jetson Orin deployments) for efficient model training.
Technology Stack
.NET 9.0 Worker Service: Provides a robust, hostable foundation for background processing.
TensorFlow.NET (SciSharp.NET): Enables native C# deep learning development with full GPU acceleration via NVIDIA CUDA. This is the core library for our ML models.
RabbitMQ.Client: Facilitates asynchronous communication by consuming CPO job requests from a message queue.
Microsoft.EntityFrameworkCore / Microsoft.Data.SqlClient: For interaction with the historical data database (via Delphi/Conductor if proxied) and persistence of optimized parameters.
Docker: For containerization, ensuring consistent environments and facilitating multi-architecture deployment (x64 and ARM64) to various compute resources.
Interaction with Other Projects
Ludus integrates seamlessly into the Temperance microservices architecture:

Conductor (API Project): Acts as the central orchestrator. After market close, or on a scheduled basis, Conductor will receive a trigger (either from Constellations' live trading conclusion or its own internal scheduler) to initiate the CPO process.
RabbitMQ (Message Queue): Conductor publishes CPO job messages to a dedicated RabbitMQ queue. These messages contain the necessary configuration for Ludus to start an optimization run (e.g., StrategyName, Symbols, LookbackPeriod).
Delphi (Historical Data Service): Ludus will retrieve required historical market data (e.g., prices from [Prices].[StockSymbol}_{Interval} tables) from the SQL Server database, likely proxied or managed through Delphi or Conductor's data access layer, to feed its machine learning models.
Constellations (Strategy Runner): Once Ludus has successfully optimized and persisted parameters, Constellations will query the database (e.g., a dedicated OptimizedParameters table) at the start of the next trading day to retrieve the latest, conditionally optimized parameters for its live trading strategies.
Observatory (Dashboard): Ludus will emit logs and status updates on its training progress and optimization results, which can be monitored and visualized in the Observatory dashboard.
Initial Implementation Workflow
Message Definition: Define a clear message contract for CPO job requests that Conductor will send and Ludus will consume (e.g., a simple JSON payload detailing the strategy, symbols, date ranges, etc.).
RabbitMQ Consumer Setup in Ludus:
Implement a BackgroundService within Ludus that listens to the designated RabbitMQ queue.
On receiving a message, it will deserialize the CPO configuration.
Data Fetching: Ludus will make calls (or use direct database access if appropriate) to retrieve historical price data for the specified symbols and periods from your SQL Server database.
TensorFlow.NET Model Integration:
Define your initial machine learning model architecture using TensorFlow.NET (or Keras.NET on top of it) within Ludus.
Implement the training logic: load data into Tensor objects, define loss functions, optimizers, and run the model.Fit() method.
Ensure CUDA and cuDNN are correctly configured for GPU acceleration on your development machine.
Parameter Persistence: After training, the optimized parameters determined by the ML model will be saved to a database table accessible by Constellations.
Basic Logging: Implement robust logging to track the progress of training, any errors, and the final optimization results.
Future Plans & Scalability
Diverse ML Models: Expand the library of machine learning models within Ludus to support various CPO approaches, from simpler regression models to complex deep neural networks.
Hyperparameter Tuning: Integrate advanced hyperparameter optimization techniques (e.g., Bayesian optimization, genetic algorithms) to find the best configuration for the ML models themselves.
A/B Testing of Optimized Parameters: Implement mechanisms to test the performance of newly optimized parameters in a simulated environment before deploying them to live strategies.
Docker Multi-Architecture Deployment: Leverage docker buildx to create multi-architecture container images (x64 for local powerful machines, ARM64 for Jetson Orin devices), enabling seamless horizontal scaling of Ludus across diverse compute infrastructure.
Dynamic Scaling: Explore orchestration tools (like Kubernetes or Docker Swarm) for dynamically scaling Ludus instances based on queue depth and processing load.
Model Versioning & Rollback: Implement a system for versioning trained models and optimized parameters, allowing for easy rollback to previous stable configurations.
Getting Started (for Developers)
To run Ludus locally:

Prerequisites:
.NET 9.0 SDK
Docker Desktop (if running via containers)
NVIDIA GPU drivers, CUDA Toolkit, and cuDNN (if leveraging GPU acceleration on your local machine). Refer to NVIDIA's documentation for installation specific to your OS.
Access to your SQL Server historical data database.
A running RabbitMQ instance.
Clone the Repository:
Bash

git clone [Your Repository URL]
cd Ludus
Install NuGet Packages:
Bash

dotnet restore
Configure appsettings.json:
Set up connection strings for your database.
Configure RabbitMQ connection details.
Run the Worker Service:
Bash

dotnet run
Alternatively, build and run as a Docker container:
Bash

docker build -t ludus-worker .
docker run -d ludus-worker
