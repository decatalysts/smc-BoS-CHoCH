# Proposed Scalable Architecture for Trading Bot

## Introduction

This document outlines a proposed scalable architecture for the trading bot. The goal is to enhance modularity, testability, maintainability, and overall scalability of the system, allowing it to grow in complexity (more strategies, more assets, more sophisticated features) and handle larger operational loads.

The current architecture, while functional, has several areas that can be improved, as noted in the main `README.md` (under "Potential Areas for Improvement"). This proposal addresses those points through a phased approach.

## Guiding Principles

*   **Separation of Concerns:** Each component should have a single, well-defined responsibility.
*   **Decoupling:** Components should interact through well-defined interfaces or an event bus, minimizing direct dependencies.
*   **Configuration Driven:** System behavior should be easily adjustable through external configurations.
*   **Testability:** Components should be designed to be easily testable in isolation and integration.
*   **Scalability:** The architecture should allow for individual components to be scaled independently if needed.

## Proposed Directory Structure (Target)

This structure supports the proposed modularity:

```
trading_bot/
├── config/                     # Configuration files (yaml, json, .env)
│   ├── __init__.py
│   └── settings.py             # Loads and provides access to config (or use a library like Pydantic)
├── core/                       # Core business logic services and components
│   ├── __init__.py
│   ├── data_ingestion_service.py # Manages market data feeds
│   ├── strategy_engine.py        # Hosts and runs trading strategies
│   ├── base_strategy.py          # Abstract base class for all strategies
│   ├── order_execution_service.py # Manages order lifecycle
│   ├── event_bus.py              # Simple event dispatcher
│   └── strategies/               # Individual strategy implementations
│       ├── __init__.py
│       ├── bos_uptrend_strategy.py
│       ├── bos_downtrend_strategy.py
│       ├── choch_uptrend_reversal_strategy.py
│       ├── choch_downtrend_reversal_strategy.py
│       └── ...                   # Other strategies
├── database/                   # Database interaction logic and models
│   ├── __init__.py
│   ├── db_controller.py        # Low-level DB connection and operations
│   └── models.py               # (Optional) Data models/schemas if using an ORM/ODM
├── indicators/                 # Technical indicators library (e.g., MyTT)
│   ├── __init__.py
│   └── MyTT.py
├── utils/                      # Common utility functions, constants
│   └── __init__.py
├── tests/                      # Unit and integration tests
│   ├── unit/
│   └── integration/
├── scripts/                    # Helper scripts (e.g., backtesting runner, deployment tools)
├── main.py                     # Main application entry point, service initialization, main loop
├── requirements.txt            # Project dependencies
├── README.md                   # Main project README
└── SCALABLE_ARCHITECTURE.md    # This document
```

## Phase 1: Core Refactoring

This phase focuses on immediate improvements to code structure and configuration management.

### 1. Centralized Configuration

*   **Objective:** Remove hardcoded parameters and centralize all configurations.
*   **Implementation:**
    *   Create a `config/settings.py` (or use YAML/JSON files loaded by `settings.py`).
    *   Move API keys (with strong recommendations for using environment variables for production), trading parameters (e.g., `max_loss_per_trade`, `wait_windows`), product lists, exchange details, database URIs, and logging settings into this configuration module.
    *   All other modules will import configurations from `config.settings`.
*   **Benefits:**
    *   Easier management of settings for different environments (development, testing, production).
    *   Reduced risk of errors from inconsistent hardcoded values.
    *   Improved security by separating sensitive data from code (especially when using environment variables).

### 2. Strategy Abstraction

*   **Objective:** Reduce code duplication in strategy files and create a clear interface for strategy development.
*   **Implementation:**
    *   Define an abstract base class `core/base_strategy.py:BaseStrategy`.
    *   `BaseStrategy` will handle:
        *   Initialization with API (or data feed), product info, config, DB controller, logger.
        *   Common data preparation: Converting k-line data to Pandas DataFrames, calculating universally used indicators (e.g., MACD, pivot points, basic SMC features like OB/FVG columns).
        *   Abstract methods to be implemented by concrete strategy classes:
            *   `_check_pattern_conditions()`: Core logic for identifying the specific pattern (BoS, CHoCH).
            *   `_find_entry_logic()`: Logic to find FVG/OB based entries once a pattern is confirmed.
            *   `_calculate_trade_parameters()`: To determine stop-loss, take-profit, position size.
        *   A public method like `run_check()` to orchestrate the strategy execution steps.
        *   Common database interaction logic for storing detected patterns (e.g., `_store_pattern()`).
    *   Refactor existing strategy files (`BoS_*.py`, `CHoCH_*.py`) into new classes (e.g., `BosUptrendStrategy`) under `core/strategies/` that inherit from `BaseStrategy`. These classes will primarily implement the abstract methods, focusing on their unique logic.
*   **Benefits:**
    *   Significant reduction in boilerplate code.
    *   Easier to develop, test, and maintain individual strategies.
    *   Consistent structure for all strategies.

### 3. Refactor Main Orchestration (`market_structures.py` -> `main.py` and `core/strategy_engine.py`)

*   **Objective:** Adapt the main application driver to use the new configuration and strategy abstraction.
*   **Implementation:**
    *   The existing `market_structures.py` logic will be split.
    *   `main.py` will be the primary entry point: initializes configurations, logging, DB connection, and the (future) services.
    *   A new `core/strategy_engine.py` will be responsible for:
        *   Loading and instantiating all strategy classes from `core/strategies/`.
        *   In its initial form (pre-Phase 2), it might still contain the main loop that iterates through products and calls `run_check()` on each strategy instance, using data directly from TqApi (passed to strategies).
*   **Benefits:**
    *   Clearer separation of application setup (`main.py`) from strategy execution logic (`strategy_engine.py`).
    *   Prepares for the introduction of the Data Ingestion Service in Phase 2.

## Phase 2: Service-Oriented Modules

This phase focuses on decoupling major components of the system into distinct services, typically communicating via an event bus.

### 1. Data Ingestion Service (`core/data_ingestion_service.py`)

*   **Objective:** Isolate market data fetching logic.
*   **Responsibilities:**
    *   Connect to the market data API (e.g., `TqApi`).
    *   Manage subscriptions to k-line data for all configured products.
    *   Fetch historical data as needed.
    *   Publish standardized "New Kline Data" events (e.g., `NewKlineEvent` containing OHLCV data) to the Event Bus.
*   **Benefits:**
    *   Strategies become independent of the specific data source API.
    *   Easier to add new data sources or modify existing ones.
    *   Centralized handling of data API specifics (rate limits, errors, reconnections).

### 2. Order Execution Service (`core/order_execution_service.py`)

*   **Objective:** Centralize and manage all trading order operations.
*   **Responsibilities:**
    *   Receive standardized "Place Order" commands (e.g., `PlaceOrderCommand`) from strategies (likely via the Event Bus or direct calls).
    *   Interact with the trading API (`TqApi`) to place, modify, or cancel orders.
    *   Track order status and lifecycle.
    *   Publish "Order Status" events (e.g., `OrderFilledEvent`, `OrderFailedEvent`) to the Event Bus.
*   **Benefits:**
    *   Strategies are decoupled from the complexities of order execution.
    *   Centralized management of exchange-specific order rules and error handling.
    *   Enables features like simulated ("paper") trading by swapping out the live execution module.

### 3. Event Bus / Dispatcher (`core/event_bus.py`)

*   **Objective:** Facilitate decoupled communication between services.
*   **Responsibilities:**
    *   Allow services to subscribe to specific event types.
    *   Allow services to publish events.
    *   Route published events to all interested subscribers.
*   **Implementation (Initial):** Can start as a simple in-process pub/sub mechanism. For larger scale, this could be replaced by a message broker (e.g., RabbitMQ, Kafka).
*   **Benefits:**
    *   Reduces direct dependencies between components.
    *   Enhances flexibility: services can be added, removed, or modified with less impact on other parts of the system.
    *   Promotes an asynchronous, event-driven flow.

### Interaction Flow (Post-Phase 2)

1.  `DataIngestionService` captures a new k-line and publishes a `NewKlineEvent`.
2.  `StrategyEngine` (listening to the Event Bus) receives the `NewKlineEvent`.
3.  `StrategyEngine` forwards the k-line data to the relevant strategy instances (e.g., `BosUptrendStrategy`).
4.  The strategy analyzes the data. If a trading signal is generated, it creates a `PlaceOrderCommand`.
5.  The strategy sends the `PlaceOrderCommand` to the `OrderExecutionService` (potentially via the Event Bus).
6.  `OrderExecutionService` processes the command, places the trade via `TqApi`.
7.  Upon trade confirmation/failure, `OrderExecutionService` publishes an `OrderFilledEvent` / `OrderFailedEvent`.
8.  Other components (e.g., a future Portfolio Management Service, or the strategy itself) can react to these order events.

## Benefits of the Proposed Architecture

*   **Improved Modularity:** Clear separation of responsibilities.
*   **Reduced Duplication:** `BaseStrategy` minimizes redundant code.
*   **Enhanced Testability:** Individual services and strategies can be unit-tested more easily. Integration testing becomes more systematic.
*   **Increased Maintainability:** Changes in one component are less likely to impact others.
*   **Better Scalability:**
    *   Services can potentially be scaled independently (e.g., run in separate processes or on different machines if using a message broker).
    *   The system can handle more strategies and products more efficiently.
*   **Flexibility:** Easier to add new strategies, data sources, or execution venues.
*   **Improved Readability:** A well-defined structure makes the codebase easier to understand.

## Advanced Scalability & Future Considerations

Beyond Phase 1 and Phase 2, further enhancements can be considered for very large-scale operations or increased functional complexity:

1.  **Dedicated Portfolio Management Service:**
    *   **Responsibility:** Track overall portfolio value, individual position P&L, risk exposure across all assets and strategies.
    *   **Benefit:** Centralized view of portfolio health, enabling more sophisticated risk management and performance attribution.

2.  **Dedicated Risk Management Service:**
    *   **Responsibility:** Implement and enforce global risk rules that supersede individual strategy risk parameters. Examples: maximum overall drawdown limits, sector exposure limits, correlation-based position limits.
    *   **Benefit:** Provides an independent layer of risk control crucial for protecting capital at scale.

3.  **Distributed Task Queues (e.g., Celery, RQ):**
    *   **Use Case:** For computationally intensive or non-time-critical tasks such as:
        *   End-of-day report generation.
        *   Complex data analysis or feature engineering for strategies.
        *   Machine learning model training/retraining (if applicable).
        *   Batch data processing.
    *   **Benefit:** Offloads work from the main trading loop, ensuring low latency for critical operations. Allows for horizontal scaling of worker processes.

4.  **Containerization (Docker) & Orchestration (Kubernetes):**
    *   **Docker:** Package the application services (Data Ingestion, Strategy Engine, Order Execution, etc.) and their dependencies into standardized containers.
        *   **Benefit:** Consistent deployment across different environments, simplified dependency management, isolation of services.
    *   **Kubernetes (or similar, like Docker Swarm):** Manage and orchestrate containerized services.
        *   **Benefit:** Automated deployment, scaling (e.g., running multiple instances of a stateless service), self-healing (restarting failed containers), load balancing, and efficient resource utilization in a clustered environment.

5.  **Advanced Monitoring, Logging, and Alerting:**
    *   **Centralized Logging:** Implement a robust centralized logging system (e.g., ELK Stack - Elasticsearch, Logstash, Kibana; or EFK Stack - Elasticsearch, Fluentd, Kibana) for aggregating and searching logs from all distributed services.
    *   **Metrics Collection & Visualization:** Use tools like Prometheus for collecting detailed metrics from all services (e.g., event processing times, queue lengths, error rates, API latencies) and Grafana for visualizing these metrics on dashboards.
    *   **Alerting:** Set up automated alerts (e.g., via Alertmanager in Prometheus, or dedicated alerting tools) for critical system failures, performance degradation, or trading anomalies.
    *   **Benefit:** Provides deep visibility into the system's health and performance, enabling proactive issue detection and faster troubleshooting.

6.  **Sophisticated Backtesting Framework:**
    *   **Features:** Event-driven backtester that can accurately simulate historical market conditions, slippage, commission, and order queue dynamics. Should allow for parameter optimization and robust performance analysis.
    *   **Benefit:** Crucial for validating strategy effectiveness and robustness before live deployment.

7.  **API-Driven Control and Extensibility:**
    *   **Internal API Gateway:** If many microservices are developed, an API gateway can manage access, routing, and authentication for internal service-to-service communication.
    *   **External Control API:** A secure API allowing external systems or UIs to monitor bot status, manage strategies (start/stop), adjust high-level parameters, or view performance reports.
    *   **Benefit:** Enables better operational control, integration with other systems, and potential for building user interfaces.

8.  **Data Storage & Analysis Scalability:**
    *   **Time-Series Databases:** For very high-volume tick data or order book data, consider specialized time-series databases (e.g., InfluxDB, TimescaleDB).
    *   **Data Warehousing/Lakes:** For large-scale historical data analysis and reporting, a data warehouse or data lake solution might be necessary.
    *   **Benefit:** Efficiently manage and analyze the vast amounts of data generated and consumed by trading operations.

These advanced options represent a significant investment in infrastructure and development but are key to building a truly enterprise-grade, highly scalable, and resilient automated trading system. The choice of which options to implement would depend on the specific growth trajectory and requirements of the trading operation.
