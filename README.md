# Trading Bot Project

## Overview

This project implements an automated trading bot that operates based on "Smart Money Concepts" (SMC) such as Break of Structure (BoS) and Change of Character (CHoCH). It connects to various futures exchanges, analyzes market data in real-time, identifies trading opportunities based on predefined patterns, and logs these events to a MongoDB database.

## Architecture

The bot follows a modular architecture:

1.  **Data Feed & Exchange Interaction:**
    *   Utilizes the `tqsdk` library (`TqApi`) to connect to exchanges and stream real-time k-line (candlestick) data for various financial products.
    *   Handles fetching historical data for initial analysis.

2.  **Signal Generation Engine:**
    *   **Pattern Detection Modules** (`BoS_Downtrend.py`, `BoS_Uptrend.py`, `CHoCH_Downtrend_Reversal.py`, `CHoCH_Uptrend_Reversal.py`): Each module is responsible for identifying a specific market structure (BoS or CHoCH for uptrends/downtrends).
    *   **Technical Analysis Library** (`MyTT.py`): Provides a comprehensive suite of technical indicators (MACD, EMA, SMA, RSI, ATR, etc.) used for pattern confirmation.
    *   **Utility Functions** (`utils.py`): Contains helper functions for custom calculations, pivot point detection, MACD divergence analysis, and other supporting logic.
    *   **Entry Logic**: Within each pattern detection module, the `find_entry` function evaluates potential entry points based on criteria like Fair Value Gaps (FVG), Order Blocks (OB), and risk/reward ratios.

3.  **Data Storage:**
    *   A MongoDB database is used to store information about detected market structures (BoS and CHoCH events).
    *   The `db/db_controller.py` module encapsulates all database interactions (connection, insertion, querying).

4.  **Orchestration & Main Loop:**
    *   The `market_structures.py` script is the main entry point and orchestrator.
    *   It initializes the API connection, sets up the list of products to monitor across different exchanges (INE, DCE, CZCE, SHFE, GFEX, CFFEX).
    *   It runs a continuous loop, waiting for k-line data updates and triggering the appropriate pattern detection functions for each product.
    *   It also configures logging for the application.

## File Structure

```
├── .gitignore                  # Specifies intentionally untracked files
├── BoS_Downtrend.py            # Logic for detecting Break of Structure in downtrends
├── BoS_Uptrend.py              # Logic for detecting Break of Structure in uptrends
├── CHoCH_Downtrend_Reversal.py # Logic for Change of Character (reversal) from downtrend
├── CHoCH_Uptrend_Reversal.py   # Logic for Change of Character (reversal) from uptrend
├── MyTT.py                     # Library for technical indicators
├── market_structures.py        # Main orchestration script, API interaction, product setup
├── utils.py                    # Utility functions for technical analysis and trading logic
├── db/
│   ├── db_controller.py        # MongoDB database interaction logic
│   └── db_test.py              # Test script for db_controller
└── logs/                       # Directory for log files (implicitly created by logger)
```

## Key Responsibilities of Files

*   **`market_structures.py`**:
    *   Initializes `TqApi` for market data.
    *   Defines product lists for exchanges (INE, DCE, CZCE, SHFE, GFEX, CFFEX).
    *   Contains the main loop that drives the strategy execution by calling detection functions.
    *   Sets up logging.
*   **`BoS_*.py`, `CHoCH_*.py` files**:
    *   Implement algorithms to detect specific market patterns (BoS, CHoCH).
    *   Contain `find_entry` functions to identify potential trade entries based on pattern confirmation (e.g., FVG, OB).
    *   Utilize `MyTT.py` and `utils.py` for calculations.
    *   Log detected patterns and potential entries.
    *   Store confirmed patterns in the MongoDB database via `db_controller`.
*   **`MyTT.py`**:
    *   A comprehensive library of common technical analysis indicators (e.g., MACD, EMA, SMA, RSI, BOLL, ATR, KDJ).
*   **`utils.py`**:
    *   Provides various helper functions:
        *   Custom technical indicator calculations (ZLEMA, Fisher Transform).
        *   Heikin Ashi candle calculations.
        *   Chandelier Exit logic.
        *   Pivot point detection (`isPivot`).
        *   MACD divergence detection (`detect_macd_divergen`).
        *   Functions to check previous pivot points for profit targets (`check_prev_pivot_point`).
*   **`db/db_controller.py`**:
    *   Manages the connection to MongoDB.
    *   Provides methods to insert and query trading data (detected market structures).
*   **`db/db_test.py`**:
    *   Unit tests for `db_controller.py`.

## Dependencies

*   `tqsdk`: For interacting with trading APIs and market data.
*   `pandas`: Used extensively for data manipulation, especially with k-line data.
*   `numpy`: For numerical operations, often used by `pandas` and `MyTT.py`.
*   `pymongo`: (Implied by `db_controller.py`) For MongoDB interaction.
*   `smartmoneyconcepts`: (Imported in BoS/CHoCH files) Likely used for identifying SMC-specific elements like Order Blocks (OB) and Fair Value Gaps (FVG).

## Potential Areas for Improvement

This section outlines potential areas where the codebase could be enhanced for better maintainability, robustness, and security.

1.  **Code Refactoring & Reducing Duplication:**
    *   **Strategy Logic:** The four strategy files (`BoS_Downtrend.py`, `BoS_Uptrend.py`, `CHoCH_Downtrend_Reversal.py`, `CHoCH_Uptrend_Reversal.py`) share a significant amount of boilerplate code for data preparation, MACD/pivot calculation, entry searching logic, and database interaction.
    *   **Recommendation:** Refactor common functionalities into a base strategy class or shared utility functions to reduce redundancy and make strategies easier to manage and modify. For example, a generic `PatternDetector` class could handle data loading, indicator calculations, and common entry search elements, with subclasses implementing specific pattern logic.

2.  **Configuration Management:**
    *   **Hardcoded Parameters:** Trading parameters (e.g., `max_loss_per_trade`, `wait_windows`), product lists for exchanges, and file paths (e.g., `"market structures/BoS_Uptrend.json"` in `BoS_Uptrend.py`) are hardcoded in various files.
    *   **Recommendation:** Externalize these configurations into a dedicated file (e.g., `config.py`, `config.json`, or YAML file) or use environment variables. This would make it easier to adjust settings without modifying the core code.

3.  **Security:**
    *   **Hardcoded Credentials:** API authentication details (`TqAuth("18822034094", "liuleren123")`) are directly embedded in `market_structures.py`. This is a major security vulnerability.
    *   **Recommendation:** Store credentials securely using environment variables, a dedicated secrets management tool, or a configuration file with strict access permissions. **This should be addressed with high priority.**

4.  **Modularity & Code Organization:**
    *   **`utils.py`:** This file contains a wide variety of functions, from specific technical indicators to general utility functions.
    *   **Recommendation:** Consider splitting `utils.py` into more focused modules (e.g., `technical_indicators.py`, `trading_helpers.py`, `data_processing.py`) to improve organization and readability.
    *   **`MyTT.py`:** If this is a custom library, ensure its functions are well-organized. If it's an external library, it should be managed as a project dependency (e.g., in a `requirements.txt` file).

5.  **Error Handling & Resilience:**
    *   The current error handling appears to be minimal. The application could be vulnerable to issues like API connection drops, unexpected data formats, or database unavailability.
    *   **Recommendation:** Implement more comprehensive error handling (try-except blocks) around API calls, database operations, and critical calculations. Add retry mechanisms for transient errors where appropriate.

6.  **Testing:**
    *   While `db/db_test.py` exists, there's a lack of unit tests for the core trading logic (pattern detection, signal generation, entry conditions).
    *   **Recommendation:** Develop a suite of unit tests using a framework like `unittest` or `pytest`. This would cover individual functions, especially in the strategy files and `utils.py`, to ensure correctness and prevent regressions. Backtesting capabilities could also be considered for strategy validation.

7.  **Readability & Internationalization:**
    *   **Mixed Languages:** There are numerous comments and some variable names in Chinese.
    *   **Recommendation:** For broader collaboration and easier maintenance, translate all comments and identifiers into English.
    *   **Code Clarity:** Some complex logic blocks, particularly in the `find_entry` functions, could benefit from more detailed English comments explaining the specific trading rules and conditions being checked.

8.  **Logging:**
    *   Logging is implemented, but it could be more structured and informative.
    *   **Recommendation:** Use different log levels (INFO, WARNING, ERROR, DEBUG) more consistently. Add more contextual information to log messages, such as the specific product and timestamp for which an event occurred. Consider using a structured logging format if logs are to be parsed by other systems.

9.  **Performance & Optimization:**
    *   The main loop in `market_structures.py` iterates through all configured products and calls detection functions upon every k-line update. This could be resource-intensive, especially with a large number of products.
    *   **Recommendation:** Profile the application to identify any performance bottlenecks. Depending on the API's capabilities, explore options like event-driven updates instead of continuous polling, or stagger the checks for different products if real-time updates for all are not strictly necessary simultaneously.

10. **Database Schema & Queries:**
    *   The structure of data stored in MongoDB isn't explicitly defined.
    *   **Recommendation:** Document the database schema. Optimize database queries if performance issues arise, ensuring appropriate indexes are in place for common query patterns.
```
