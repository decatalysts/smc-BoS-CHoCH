from abc import ABC, abstractmethod
import pandas as pd
from MyTT import MACD, SMA, EMA, REF, LLV, HHV # Assuming MyTT is in the python path or project root
from utils import isPivot, detect_macd_divergen # Assuming utils.py is in the python path or project root
from smartmoneyconcepts import smc # Assuming this library is installed and importable

class BaseStrategy(ABC):
    """
    Abstract Base Class for trading strategies.

    This class provides a foundational structure for developing various trading strategies.
    It handles common tasks such as:
    - Preparing market data by converting raw k-line data (e.g., from TQSDK) into
      a pandas DataFrame.
    - Calculating a standard set of technical indicators (e.g., MACD, pivot points)
      and integrating features from Smart Money Concepts (SMC) like Order Blocks (OB)
      and Fair Value Gaps (FVG).
    - Defining an execution flow for strategies, including checking for new patterns,
      identifying entry points, calculating trade parameters, and storing pattern data.
    - Interacting with a database for storing and retrieving pattern information.
    - Logging strategy actions and potential trade signals.

    Concrete strategy implementations should inherit from this class and implement
    the abstract methods to define their specific pattern detection logic,
    entry point identification rules, and trade parameter calculations.
    """

    def __init__(self, api, product_info: dict, config: object, db_controller: object, logger: object):
        """
        Initializes the BaseStrategy.

        Args:
            api (object): Instance of TqApi or a compatible API wrapper used for market data
                          and potentially trade execution.
            product_info (dict): A dictionary containing product-specific details.
                Expected keys:
                'kline_data' (object): Raw k-line data provider (e.g., TQSDK KlineSerializer).
                'product_symbol' (str): User-friendly symbol for the product (e.g., 'sc', 'MA').
                'exchange_symbol' (str): Symbol of the exchange (e.g., 'INE', 'CZCE').
                'underlying_symbol' (str): The symbol used for API calls to TQSDK or other brokers.
                'quote' (object): TQSDK quote object or similar for the product. (Currently unused in BaseStrategy but good for context)
            config (object): A configuration object or module providing access to trading
                             parameters, system settings (e.g., PIVOT_WINDOW, SMC_SWING_LENGTH,
                             WAIT_WINDOWS, KLINE_FREQ_SECONDS), and potentially API keys.
            db_controller (object): An instance of a database controller class, providing methods
                                    like `find_by_condition` and `insert` for database interactions.
            logger (object): An instance of a logger (e.g., from Python's `logging` module)
                             for recording information, warnings, and errors.
        """
        self.api = api
        self.product_info = product_info
        self.config = config
        self.db = db_controller
        self.logger = logger

        self.kline_data_raw = product_info['kline_data']
        self.product_symbol = product_info['product_symbol']  # User-friendly name, e.g., 'sc', 'MA'
        self.exchange_symbol = product_info['exchange_symbol']  # e.g., 'INE', 'CZCE'
        self.underlying_symbol = product_info['underlying_symbol']  # Symbol for TQSDK API calls

        self.df: pd.DataFrame = None  # Stores the processed k-line data with indicators
        self.current_row: pd.Series = None  # The most recent complete candle's data
        self.current_datetime_str: str = None  # String representation of the current candle's datetime

        # These attributes must be defined by concrete strategy classes if they intend to use
        # the _check_historical_patterns_for_entry method.
        # self.db_collection_for_historical_check: Name of the DB collection for this strategy's patterns.
        # self.datetime_field_for_historical_query: The specific datetime field in the stored pattern
        #                                           that represents the pattern's occurrence time
        #                                           (e.g., 'break_datetime', 'reverse_datetime').
        self.db_collection_for_historical_check: str = None
        self.datetime_field_for_historical_query: str = None


    def _prepare_data(self) -> bool:
        """
        Prepares and enriches the k-line data for analysis.

        This method performs several key steps:
        1. Checks if there's new k-line data available using `api.is_changing()`.
        2. Converts the raw k-line data (excluding the latest, possibly incomplete, candle)
           into a pandas DataFrame.
        3. Sets a pandas DatetimeIndex and ensures a 'datetime' column is present.
        4. Calculates common technical indicators:
           - MACD (Moving Average Convergence Divergence).
           - MACD divergence using `detect_macd_divergen` from `utils.py`.
           - Pivot points using `isPivot` from `utils.py`.
        5. Integrates Smart Money Concepts (SMC) features:
           - Swing Highs/Lows.
           - Order Blocks (OB).
           - Fair Value Gaps (FVG).
           This involves preparing a temporary DataFrame for the `smc` library,
           handling potential NaNs, and merging the results back.
        6. Updates `self.df` with the processed DataFrame and `self.current_row`
           with the data of the most recent complete candle.
        7. Sets `self.current_datetime_str` to the datetime of the current candle.

        Returns:
            bool: True if new data was successfully prepared and `self.df` is updated,
                  False otherwise (e.g., no new data, insufficient data, or an error occurred).
        """
        # Check if the last candle's datetime has changed, indicating a new candle might have formed
        if not self.api.is_changing(self.kline_data_raw.iloc[-1], "datetime"):
            return False # No new tick or data update relevant to kline formation

        try:
            # Store the datetime of the latest raw kline data point as a string
            # This is used as 'detect_datetime' when storing patterns
            self.current_datetime_str = pd.to_datetime(
                self.kline_data_raw.iloc[-1]['datetime'], unit='ns'
            ).strftime('%Y-%m-%d %H:%M:%S')

            # Convert TQSDK kline series to a list of dictionaries,
            # excluding the last (potentially incomplete) candle.
            # The strategy should only operate on closed/complete candles.
            records = []
            for i in range(len(self.kline_data_raw) - 1):  # Exclude the last candle
                records.append(self.kline_data_raw.iloc[i].to_dict())
            
            if not records:
                self.logger.info(f"[{self.product_symbol}] No complete candles available to process yet.")
                return False

            df = pd.DataFrame(records)
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ns')
            # Set DatetimeIndex for time-series operations, but keep 'datetime' as a column for convenience
            df = df.set_index(pd.DatetimeIndex(df['datetime']), drop=False)

            # --- Calculate Common Technical Indicators ---
            df['DIFF'], df['DEA'], df['MACD'] = MACD(df['close'])
            df = detect_macd_divergen(df)  # Expects df and adds divergence columns

            # Calculate pivot points; isPivot expects the index and window size
            # PIVOT_WINDOW needs to be defined in self.config
            df['is_pivot'] = [isPivot(idx, self.config.PIVOT_WINDOW, df) for idx in range(len(df))]

            # --- Integrate Smart Money Concepts (SMC) Features ---
            # The `smc` library might expect specific column names (e.g., 'Open', 'High', 'Low', 'Close', 'Volume')
            # and might be sensitive to NaNs in input columns.
            # Create a clean DataFrame copy for `smc` library, ensuring correct column names.
            smc_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            smc_df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            }, inplace=True)
            
            # Ensure no NaNs in core OHLCV columns for the `smc` library if it's sensitive.
            smc_df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
            
            if not smc_df.empty:
                # Calculate swing highs and lows using the smc library
                # SMC_SWING_LENGTH needs to be defined in self.config
                swing_highs_lows = smc.swing_highs_lows(smc_df, swing_length=self.config.SMC_SWING_LENGTH)
                
                # Order Blocks (OB)
                if not swing_highs_lows.empty:
                    # Calculate Order Blocks; `close_mitigation=False` means OBs are not considered mitigated by wicks.
                    ob_data = smc.ob(smc_df, swing_highs_lows=swing_highs_lows, close_mitigation=False)
                    # Merge OB data back to the main DataFrame. Use `join` as smc results are indexed like smc_df.
                    # `ob_data` might return columns like 'OB', 'High', 'Low', 'Mitigation_Price', 'Direction'.
                    # We are interested in the 'OB' column which likely indicates the strength or presence of an OB.
                    df = df.join(ob_data[['OB']]) # Ensure only desired columns are joined
                else:
                    df['OB'] = pd.NA # Or appropriate default if no swings (e.g., 0 or NaN)

                # Fair Value Gaps (FVG)
                # `join_consecutive=False` means consecutive FVGs are treated as separate instances.
                fvg_data = smc.fvg(smc_df, join_consecutive=False)
                # Merge FVG data. `fvg_data` might return 'FVG', 'Top', 'Bottom', 'Mitigated'.
                # 'FVG' often indicates the size of the gap or a boolean presence.
                # 'Top' and 'Bottom' define the FVG range.
                df = df.join(fvg_data[['FVG', 'Top', 'Bottom']])
            else:
                self.logger.warning(f"[{self.product_symbol}] SMC DataFrame is empty after cleaning. Not enough data for SMC calculations.")
                # Ensure columns exist even if SMC calculations are skipped
                df['OB'] = pd.NA
                df['FVG'] = pd.NA
                df['Top'] = pd.NA
                df['Bottom'] = pd.NA

            self.df = df
            if not self.df.empty:
                self.current_row = self.df.iloc[-1]  # Set current_row to the last complete candle
            else:
                # This case should ideally not be reached if `records` were present earlier.
                self.logger.warning(f"[{self.product_symbol}] DataFrame became empty after processing. This is unexpected.")
                return False

            return True # Data preparation successful

        except Exception as e:
            self.logger.error(f"[{self.product_symbol}] Error in _prepare_data: {e}", exc_info=True)
            self.df = None  # Invalidate DataFrame on error to prevent use of stale/corrupt data
            return False

    @abstractmethod
    def _check_pattern_conditions(self) -> dict | None:
        """
        Analyzes `self.df` to identify the specific market pattern defined by the subclass.

        This method must be implemented by concrete strategy subclasses. It should contain
        the core logic for detecting the targeted trading pattern (e.g., Break of Structure,
        Change of Character) based on the data in `self.df`.

        Returns:
            dict | None: If the pattern is detected, this method should return a dictionary
                         containing key details about the pattern. This dictionary must include:
                         - 'db_collection_name' (str): The name of the MongoDB collection
                           where this type of pattern should be stored.
                         - 'datetime_field_for_query' (str): The name of the specific datetime
                           field within the pattern dictionary that uniquely identifies the
                           pattern's timing (e.g., 'break_datetime', 'reverse_datetime'). This
                           is used for querying existing patterns.
                         - Any other pattern-specific information (e.g., 'break_price',
                           'direction', 'pivot_points_involved').
                         The unique datetime field itself (e.g. 'break_datetime') must also be
                         a key in the returned dictionary, holding the actual datetime value.
                         If the pattern is not detected, this method should return None.
        """
        pass

    @abstractmethod
    def _find_entry_points(self, pattern_details: dict) -> list[dict] | None:
        """
        Identifies potential entry points based on the detected pattern and SMC features.

        This method must be implemented by concrete strategy subclasses. After a pattern
        is confirmed by `_check_pattern_conditions`, this method is called to find
        suitable entry points, often leveraging SMC concepts like Fair Value Gaps (FVG)
        or Order Blocks (OB) present in `self.df` and `self.current_row`.

        Args:
            pattern_details (dict): The dictionary containing details of the pattern
                                    as returned by `_check_pattern_conditions`.

        Returns:
            list[dict] | None: A list of dictionaries, where each dictionary represents a
                               potential entry point. Each entry point dictionary should
                               contain details such as:
                               - 'type' (str): The type of entry (e.g., 'FVG_entry', 'OB_entry').
                               - 'ideal_entry_price' (float): The calculated ideal price for entry.
                               - 'entry_range_high' (float): Upper bound of the entry zone.
                               - 'entry_range_low' (float): Lower bound of the entry zone.
                               - Other relevant details (e.g., 'fvg_top', 'ob_level').
                               If no suitable entry points are found, returns None or an empty list.
        """
        pass

    @abstractmethod
    def _calculate_trade_parameters(self, entry_point: dict, pattern_details: dict) -> dict | None:
        """
        Calculates trade parameters (stop-loss, take-profit, position size) for a given entry.

        This method must be implemented by concrete strategy subclasses. It takes an
        identified entry point and the original pattern details to determine appropriate
        risk management parameters.

        Args:
            entry_point (dict): A dictionary detailing the chosen entry point, as returned
                                by `_find_entry_points`.
            pattern_details (dict): The dictionary for the detected pattern, as returned
                                    by `_check_pattern_conditions`.

        Returns:
            dict | None: A dictionary containing the calculated trade parameters. Expected keys include:
                         - 'stop_loss' (float): The calculated stop-loss price.
                         - 'take_profit' (float): The calculated take-profit price.
                         - 'profit_factor' (float): The risk/reward ratio (take_profit_pips / stop_loss_pips).
                         - 'lots' (float): The position size for the trade.
                         If parameters cannot be determined or the trade is deemed invalid
                         (e.g., poor risk/reward), returns None.
        """
        pass

    def _store_pattern(self, pattern_details: dict) -> None:
        """
        Stores the detected pattern in the database if it's not already present.

        This method checks if a pattern with the same product symbol and unique pattern datetime
        already exists in the specified database collection. If not, it inserts the new
        pattern details, along with the product symbol and the detection timestamp.

        Args:
            pattern_details (dict): A dictionary containing the details of the detected pattern.
                                    Must include 'db_collection_name' and a unique datetime field
                                    (specified by 'datetime_field_for_query') for identification.
        """
        if not pattern_details:
            self.logger.debug(f"[{self.product_symbol}] Attempted to store an empty pattern. Skipping.")
            return

        collection_name = pattern_details.get('db_collection_name')
        # The unique datetime field for this pattern type (e.g., 'break_datetime')
        datetime_field_key = pattern_details.get('datetime_field_for_query')
        # The actual datetime value of when the pattern occurred
        unique_datetime_value = pattern_details.get(datetime_field_key)

        if not collection_name or not datetime_field_key or not unique_datetime_value:
            self.logger.warning(
                f"[{self.product_symbol}] Pattern details are missing essential fields for DB storage. "
                f"Need 'db_collection_name', 'datetime_field_for_query', and its corresponding value. "
                f"Provided: collection_name='{collection_name}', datetime_field_key='{datetime_field_key}', "
                f"unique_datetime_value='{unique_datetime_value}'."
            )
            return

        try:
            # Prepare condition to check for existing pattern
            condition = {
                'product': self.product_symbol,
                datetime_field_key: unique_datetime_value # Query using the pattern's specific datetime field
            }
            hist_pattern = self.db.find_by_condition(collection_name, condition)

            if not hist_pattern:
                # Pattern does not exist, so store it
                pattern_to_store = pattern_details.copy()
                pattern_to_store['product'] = self.product_symbol
                # `self.current_datetime_str` is when this kline event was processed by _prepare_data
                pattern_to_store['detect_datetime'] = self.current_datetime_str
                
                # Ensure all datetime objects within the pattern_details are converted to strings
                # before database insertion, to maintain a consistent format.
                for key, value in pattern_to_store.items():
                    if isinstance(value, pd.Timestamp):
                        pattern_to_store[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(value, list) and value and isinstance(value[0], pd.Timestamp):
                         # Handle lists of timestamps, e.g., for multiple pivot points
                         pattern_to_store[key] = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in value]

                self.db.insert(collection_name, pattern_to_store)
                self.logger.info(
                    f"[{self.product_symbol}] Stored new '{collection_name}' pattern "
                    f"with '{datetime_field_key}' at {unique_datetime_value}."
                )
            # else:
                # self.logger.debug(f"[{self.product_symbol}] Pattern {collection_name} for {datetime_field_key} at {unique_datetime_value} already exists. Not storing.")
        except Exception as e:
            self.logger.error(f"[{self.product_symbol}] Error storing pattern for {collection_name}: {e}", exc_info=True)

    def _process_entry_attempts(self, entry_point: dict, trade_parameters: dict, pattern_details: dict) -> None:
        """
        Logs detailed information about a potential trade signal.

        In a more advanced architecture, this method would be responsible for
        triggering an order placement request to an Order Execution Service.
        For now, it logs the signal details with a "TRADE_SIGNAL" prefix.

        Args:
            entry_point (dict): Dictionary detailing the entry point (e.g., type, price).
            trade_parameters (dict): Dictionary with trade parameters (stop_loss, take_profit, lots).
            pattern_details (dict): Dictionary of the pattern that triggered this entry.
        """
        if not entry_point or not trade_parameters or not pattern_details:
            self.logger.warning(f"[{self.product_symbol}] Attempted to process entry with incomplete details. Skipping.")
            return
        
        # Determine trade direction from pattern_details (subclass should provide 'direction' key)
        direction = pattern_details.get('direction', 'UNKNOWN_DIRECTION')
        pattern_type = pattern_details.get('pattern_type', 'N/A') # Subclass should provide 'pattern_type'
        entry_type = entry_point.get('type', 'N/A')
        # Ensure 'ideal_entry_price' is present, fallback to current close if not (though it should be)
        ideal_entry_price = entry_point.get('ideal_entry_price', self.current_row['close'] if self.current_row is not None else 'N/A')
        
        # The unique datetime identifier for the pattern (e.g., 'break_datetime', 'reverse_datetime')
        pattern_datetime_field = pattern_details.get('datetime_field_for_query', 'N/A_datetime_field')
        pattern_time = pattern_details.get(pattern_datetime_field, 'N/A_pattern_time')


        self.logger.warning( # Using warning level for high visibility of trade signals
            f"TRADE_SIGNAL for [{self.product_symbol}]: "
            f"Pattern: {pattern_type}, "
            f"EntryType: {entry_type}, "
            f"Direction: {direction}, "
            f"IdealEntry: {ideal_entry_price}, " # Format price if it's a float
            f"StopLoss: {trade_parameters['stop_loss']:.5f}, "
            f"TakeProfit: {trade_parameters['take_profit']:.5f}, "
            f"Lots: {trade_parameters['lots']}, "
            f"ProfitFactor: {trade_parameters['profit_factor']:.2f}, "
            f"SignalTime: {self.current_datetime_str}, " # When the signal was generated by the bot
            f"PatternTime: {pattern_time}" # When the actual pattern occurred on chart
        )

    def _check_historical_patterns_for_entry(self) -> bool:
        """
        Checks for actionable entries based on recently stored historical patterns.

        This method queries the database for patterns of this strategy's type
        (defined by `self.db_collection_for_historical_check`) that have occurred
        within a defined `WAIT_WINDOWS` look-back period. If such patterns are found,
        it attempts to find entry points using the strategy's `_find_entry_points`
        and `_calculate_trade_parameters` methods.

        Returns:
            bool: True if a trade attempt was made based on a historical pattern,
                  False otherwise.
        """
        # These attributes must be set by the concrete strategy class
        if not self.db_collection_for_historical_check or not self.datetime_field_for_historical_query:
            self.logger.debug(
                f"[{self.product_symbol}] Historical check skipped: "
                f"'db_collection_for_historical_check' or 'datetime_field_for_historical_query' not set."
            )
            return False

        # Calculate the look_back datetime limit based on WAIT_WINDOWS from config
        # KLINE_FREQ_SECONDS should be in config, representing seconds per k-line bar (e.g., 60 for 1-min, 300 for 5-min)
        try:
            # WAIT_WINDOWS and KLINE_FREQ_SECONDS need to be defined in self.config
            wait_window_seconds = self.config.WAIT_WINDOWS * self.config.KLINE_FREQ_SECONDS
        except AttributeError as e:
            self.logger.error(f"[{self.product_symbol}] Missing WAIT_WINDOWS or KLINE_FREQ_SECONDS in config: {e}. Cannot check historical patterns.")
            return False
            
        wait_window_duration = pd.Timedelta(seconds=wait_window_seconds)
        current_event_time = pd.to_datetime(self.current_datetime_str) # Time of the current k-line event
        look_back_datetime_limit = current_event_time - wait_window_duration
        
        # Construct the query for historical patterns
        # The datetime field in the DB should be stored in a queryable format (e.g., ISODate or string 'YYYY-MM-DD HH:MM:SS')
        query = {
            'product': self.product_symbol,
            # Query for patterns whose specific occurrence time is within the look-back window
            self.datetime_field_for_historical_query: {'$gte': look_back_datetime_limit.strftime('%Y-%m-%d %H:%M:%S')}
        }
        
        try:
            historical_patterns = self.db.find_by_condition(self.db_collection_for_historical_check, query)
            if not historical_patterns:
                # self.logger.debug(f"[{self.product_symbol}] No recent historical patterns found in {self.db_collection_for_historical_check} since {look_back_datetime_limit.strftime('%Y-%m-%d %H:%M:%S')}.")
                return False

            for hist_pattern in historical_patterns:
                # Ensure the pattern's time is not too old (already covered by $gte)
                # and, crucially, not in the future relative to the current candle's time.
                # Also, ensure it's not the exact same candle that formed the pattern initially if this check runs immediately.
                pattern_occurrence_time_str = hist_pattern.get(self.datetime_field_for_historical_query)
                if not pattern_occurrence_time_str:
                    self.logger.warning(f"[{self.product_symbol}] Historical pattern missing datetime field '{self.datetime_field_for_historical_query}'. Skipping.")
                    continue
                
                pattern_time = pd.to_datetime(pattern_occurrence_time_str)

                # Filter: Pattern must be before the current candle's time.
                # And pattern must be within the valid look_back window.
                if not (look_back_datetime_limit <= pattern_time < current_event_time):
                    # self.logger.debug(f"[{self.product_symbol}] Historical pattern at {pattern_time} is outside active window ({look_back_datetime_limit} to {current_event_time}). Skipping.")
                    continue

                # --- Add specific pre-filtering logic to avoid re-processing stale signals ---
                # This is a critical part to prevent redundant checks or entries on patterns
                # where the market has already moved significantly away from the ideal entry conditions.
                # Example: For a BoS Downtrend, if current low is already below the pattern's break_low,
                # it might be too late or the structure is continuing without a pullback.
                # These conditions are strategy-specific and should ideally be part of `_find_entry_points`
                # or a dedicated pre-filter method in the subclass.
                # The placeholder conditions below are illustrative and depend on keys existing in `hist_pattern`.
                current_low = self.current_row['low']
                current_high = self.current_row['high']
                pattern_type = hist_pattern.get('pattern_type') # Subclass should ensure this key exists in stored patterns

                if pattern_type == 'bos_downtrend' and current_low <= hist_pattern.get('break_low', -float('inf')):
                    # self.logger.debug(f"[{self.product_symbol}] Historical BoS Downtrend ({pattern_time}): Current low ({current_low}) already below break_low. Skipping entry check.")
                    continue
                if pattern_type == 'bos_uptrend' and current_high >= hist_pattern.get('break_high', float('inf')):
                    # self.logger.debug(f"[{self.product_symbol}] Historical BoS Uptrend ({pattern_time}): Current high ({current_high}) already above break_high. Skipping entry check.")
                    continue
                # Similar checks for CHoCH patterns (e.g., if price has moved far beyond reversal_high/low)
                if pattern_type == 'choch_downtrend_reversal' and current_high >= hist_pattern.get('reverse_high', float('inf')): # Example condition
                    continue
                if pattern_type == 'choch_uptrend_reversal' and current_low <= hist_pattern.get('reverse_low', -float('inf')): # Example condition
                    continue
                
                # If the pattern passes initial filters, try to find entry points.
                # `hist_pattern` (from DB) should contain all necessary fields that `_find_entry_points` expects.
                entry_points = self._find_entry_points(hist_pattern)
                if entry_points:
                    # Typically, process the first valid entry point found.
                    # More sophisticated logic could rank or filter these entry_points.
                    entry_point_to_process = entry_points[0] 
                    
                    trade_params = self._calculate_trade_parameters(entry_point_to_process, hist_pattern)
                    if trade_params:
                        self.logger.info(
                            f"[{self.product_symbol}] Processing entry for HISTORICAL pattern: "
                            f"{pattern_type} from {pattern_time.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        self._process_entry_attempts(entry_point_to_process, trade_params, hist_pattern)
                        return True # Signal processed from a historical pattern, exit loop for this cycle
            
            return False # No actionable entry found from historical patterns in this cycle
        except Exception as e:
            self.logger.error(f"[{self.product_symbol}] Error checking historical patterns: {e}", exc_info=True)
            return False


    def run_check(self) -> None:
        """
        Orchestrates the strategy execution cycle for the current k-line data.

        This is the main public method called externally to run the strategy logic.
        The process is as follows:
        1. Calls `_prepare_data()` to process the latest k-line data and calculate indicators.
           If data preparation fails or there's no new data, the method returns early.
        2. Calls `_check_pattern_conditions()` (implemented by the subclass) to detect
           if the strategy's specific trading pattern has occurred on the latest data.
        3. If a new pattern is detected:
           a. Calls `_store_pattern()` to save the pattern details to the database.
           b. Calls `_find_entry_points()` (subclass) to identify potential trade entries for this new pattern.
           c. If entry points are found, takes the first one and calls `_calculate_trade_parameters()` (subclass).
           d. If valid trade parameters are returned, calls `_process_entry_attempts()` to log/signal the trade.
           e. Sets `live_trade_attempted` to True.
        4. If no trade signal was generated from a live, new pattern (`live_trade_attempted` is False),
           then calls `_check_historical_patterns_for_entry()` to see if any recently stored
           patterns (within `WAIT_WINDOWS`) now offer a valid entry opportunity.
        5. If a live trade was processed, the historical check is skipped for this cycle to avoid
           potential conflicting signals from the same k-line event.
        """
        if not self._prepare_data():
            # No new complete candle, or an error occurred during data preparation.
            # _prepare_data logs reasons for failure internally.
            return

        live_trade_attempted = False
        # Check for patterns forming on the most recent complete candle
        pattern_details = self._check_pattern_conditions()

        if pattern_details:
            # A pattern specific to the inheriting strategy was found on the current data.
            self._store_pattern(pattern_details) # Store it if new
            
            # Now, try to find an entry for this freshly detected pattern
            entry_points = self._find_entry_points(pattern_details)
            if entry_points:
                # For simplicity, taking the first entry point.
                # A more robust system might rank or filter these entry_points.
                entry_point_to_process = entry_points[0] 
                
                trade_parameters = self._calculate_trade_parameters(entry_point_to_process, pattern_details)
                if trade_parameters:
                    self.logger.info(
                        f"[{self.product_symbol}] Processing entry for LIVE pattern: "
                        f"{pattern_details.get('pattern_type', 'N/A')} at {self.current_datetime_str}"
                    )
                    self._process_entry_attempts(entry_point_to_process, trade_parameters, pattern_details)
                    live_trade_attempted = True # Mark that a trade based on live pattern was processed
        
        # If no trade was initiated based on a pattern detected on the *current* candle,
        # then check if any *recent historical* patterns now offer an entry opportunity.
        if not live_trade_attempted:
            # self.logger.debug(f"[{self.product_symbol}] No live trade signal. Checking for historical pattern entries.")
            self._check_historical_patterns_for_entry()
        else:
            self.logger.info(
                f"[{self.product_symbol}] Live trade signal was processed. "
                f"Skipping historical pattern check for this cycle to prevent duplicate signals on the same candle."
            )
# Testing Strategy for BaseStrategy
#
# Framework: pytest
# Mocking: unittest.mock.patch or pytest-mock (mocker fixture)
#
# 1. Testing __init__:
#    - Purpose: Verify that all attributes are correctly initialized from constructor arguments.
#    - Method:
#        - Create mock objects for `api`, `db_controller`, `logger`, and `config`.
#        - Create a `product_info` dictionary.
#        - Instantiate `BaseStrategy` (or a minimal concrete subclass for testing abstract class instantiation).
#        - Assert that `self.api`, `self.product_info`, `self.config`, `self.db`, `self.logger`,
#          `self.kline_data_raw`, `self.product_symbol`, etc., are set to the mocked/provided values.
#        - Assert that `self.df`, `self.current_row`, `self.current_datetime_str` are None initially.
#        - Assert `self.db_collection_for_historical_check` and `self.datetime_field_for_historical_query` are None.
#
# 2. Testing _prepare_data():
#    - Purpose: Verify data transformation, indicator calculation, SMC integration, and state updates.
#    - Mocks:
#        - Mock `self.api.is_changing` to control execution flow (return True or False).
#        - Mock `self.kline_data_raw` (e.g., a MagicMock behaving like TQSDK KlineSerializer or a list of dicts).
#        - Mock `self.logger` to check for logged messages.
#        - Mock `MyTT` functions (MACD) and `utils` functions (isPivot, detect_macd_divergen)
#          to return predefined values or verify they are called with correct DataFrames.
#          Alternatively, for core indicators like MACD, use small, known datasets and verify outputs.
#        - Mock `smartmoneyconcepts.smc` functions (`swing_highs_lows`, `ob`, `fvg`) to return predefined
#          DataFrames or verify calls, as their internal logic can be complex.
#        - Mock `self.config` to provide necessary parameters (PIVOT_WINDOW, SMC_SWING_LENGTH).
#    - Scenarios:
#        - `api.is_changing` returns `False`: Method should return `False`, `self.df` remains unchanged/None.
#        - Empty `self.kline_data_raw` (after excluding last): Should log info and return `False`.
#        - Insufficient data for indicators (e.g., too few rows for MACD):
#          - If indicators raise errors, test graceful handling (error logged, returns `False`, `self.df` is None).
#          - If indicators return NaNs, test that `self.df` contains these NaNs.
#        - "Happy path" with sufficient valid data:
#            - Verify method returns `True`.
#            - Verify `self.current_datetime_str` is correctly formatted from the last raw kline.
#            - Verify `self.df` is a pandas DataFrame with expected columns (OHLC, datetime, MACD columns,
#              is_pivot, OB, FVG, Top, Bottom).
#            - For a small, controlled input `kline_data_raw`, manually calculate expected MACD values,
#              pivot points, and simple SMC features (if feasible, otherwise rely on mocking smc outputs)
#              and assert `self.df` contents match.
#            - Verify `self.current_row` is the last row of the processed `self.df`.
#        - `smc_df` becomes empty after dropping NaNs (if OHLCV has NaNs):
#            - Verify warning is logged.
#            - Verify SMC columns ('OB', 'FVG', 'Top', 'Bottom') are present in `self.df` but filled with pd.NA.
#        - Exception during any calculation (e.g., MACD call fails):
#            - Verify error is logged via `self.logger.error`.
#            - Verify method returns `False`.
#            - Verify `self.df` is set to `None`.
#
# 3. Testing Abstract Methods (_check_pattern_conditions, _find_entry_points, _calculate_trade_parameters):
#    - Statement: These methods are abstract and contain no logic in `BaseStrategy`.
#      They are tested through the implementations in concrete strategy subclasses.
#      No direct unit tests for these in `BaseStrategy_test.py`.
#
# 4. Testing _store_pattern():
#    - Purpose: Verify correct database interaction for storing new patterns and skipping duplicates.
#    - Mocks:
#        - Mock `self.db.find_by_condition`.
#        - Mock `self.db.insert`.
#        - Mock `self.logger`.
#    - Scenarios:
#        - `pattern_details` is None or empty: Method should return early, no DB calls.
#        - `pattern_details` missing 'db_collection_name' or 'datetime_field_for_query' or its value:
#            - Log warning.
#            - No DB calls.
#            - Method returns.
#        - New pattern ( `self.db.find_by_condition` returns None/empty list):
#            - `self.db.insert` should be called once.
#            - Verify the dictionary passed to `insert` contains all original `pattern_details`
#              plus 'product': `self.product_symbol` and 'detect_datetime': `self.current_datetime_str`.
#            - Verify datetimes in `pattern_to_store` are converted to strings.
#            - Info message logged.
#        - Existing pattern (`self.db.find_by_condition` returns a pattern):
#            - `self.db.insert` should NOT be called.
#            - (Optional) Debug message logged if that behavior is desired.
#        - Exception during `self.db.find_by_condition` or `self.db.insert`:
#            - Error message logged.
#
# 5. Testing _process_entry_attempts():
#    - Purpose: Verify that trade signal information is logged correctly.
#    - Mocks:
#        - Mock `self.logger.warning`.
#        - `self.current_row` might need to be a mock/dummy Series if 'close' is accessed as fallback.
#    - Scenarios:
#        - `entry_point`, `trade_parameters`, or `pattern_details` is None/empty:
#            - Warning logged about incomplete details.
#            - Main log message not generated.
#        - Valid inputs:
#            - `self.logger.warning` called once.
#            - Verify the log message string contains all the expected fields (Pattern, EntryType,
#              Direction, IdealEntry, StopLoss, TakeProfit, Lots, ProfitFactor, SignalTime, PatternTime)
#              and that their values are correctly formatted from the input dictionaries.
#
# 6. Testing _check_historical_patterns_for_entry():
#    - Purpose: Verify logic for querying historical patterns and processing potential entries from them.
#    - Mocks:
#        - Mock `self.config` to provide `WAIT_WINDOWS` and `KLINE_FREQ_SECONDS`.
#        - Mock `self.db.find_by_condition` to return:
#            - Empty list (no recent patterns).
#            - List of mock historical patterns (dictionaries).
#        - Mock `self._find_entry_points` (as it's abstract).
#        - Mock `self._calculate_trade_parameters` (as it's abstract).
#        - Mock `self._process_entry_attempts`.
#        - Mock `self.logger`.
#        - `self.current_row` needs to be a valid Series for filtering logic.
#        - Set `self.db_collection_for_historical_check` and `self.datetime_field_for_historical_query`.
#    - Scenarios:
#        - `db_collection_for_historical_check` or `datetime_field_for_historical_query` not set:
#            - Method returns `False` early. Debug log.
#        - `WAIT_WINDOWS` or `KLINE_FREQ_SECONDS` missing in config: Error log, returns `False`.
#        - No recent historical patterns found (`self.db.find_by_condition` returns empty):
#            - Method returns `False`. (Optional) Debug log.
#        - Historical patterns found, but:
#            - Pattern time is outside the (< current_event_time) check: Loop continues or finishes, returns `False`.
#            - Pattern time is too old (before look_back_datetime_limit): Loop continues/finishes, `False`. (Covered by query but good to double check logic if any)
#            - Strategy-specific pre-filtering condition (e.g., `current_low <= hist_pattern['break_low']`) is met:
#                - Loop continues to next pattern. If all filtered out, returns `False`.
#            - `_find_entry_points` returns None for a pattern: Loop continues.
#            - `_calculate_trade_parameters` returns None for an entry: Loop continues.
#        - Valid historical pattern leads to entry:
#            - `self.db.find_by_condition` returns a suitable pattern.
#            - `_find_entry_points` returns a list with at least one entry dict.
#            - `_calculate_trade_parameters` returns a valid params dict.
#            - `self._process_entry_attempts` is called with the correct arguments.
#            - Method returns `True`.
#        - Exception during DB query or processing: Error logged, returns `False`.
#
# 7. Testing run_check():
#    - Purpose: Verify the orchestration of methods within `run_check`. This is more of an
#      integration test for the class's internal workflow.
#    - Mocks:
#        - Mock `self._prepare_data`.
#        - Mock `self._check_pattern_conditions` (abstract).
#        - Mock `self._store_pattern`.
#        - Mock `self._find_entry_points` (abstract).
#        - Mock `self._calculate_trade_parameters` (abstract).
#        - Mock `self._process_entry_attempts`.
#        - Mock `self._check_historical_patterns_for_entry`.
#    - Scenarios:
#        - `_prepare_data` returns `False`: Method returns early, no other methods called.
#        - `_prepare_data` returns `True`:
#            - `_check_pattern_conditions` returns `None` (no live pattern):
#                - `_store_pattern`, `_find_entry_points` (for live), `_calculate_trade_parameters` (for live),
#                  `_process_entry_attempts` (for live) are NOT called.
#                - `_check_historical_patterns_for_entry` IS called.
#            - `_check_pattern_conditions` returns `pattern_details`:
#                - `_store_pattern` IS called with `pattern_details`.
#                - `_find_entry_points` IS called with `pattern_details`.
#                - If `_find_entry_points` returns entries:
#                    - `_calculate_trade_parameters` IS called.
#                    - If `_calculate_trade_parameters` returns params:
#                        - `_process_entry_attempts` IS called.
#                        - `_check_historical_patterns_for_entry` is NOT called.
#                    - If `_calculate_trade_parameters` returns `None`:
#                        - `_process_entry_attempts` (for live) is NOT called.
#                        - `_check_historical_patterns_for_entry` IS called.
#                - If `_find_entry_points` returns `None`:
#                    - `_calculate_trade_parameters` and `_process_entry_attempts` (for live) are NOT called.
#                    - `_check_historical_patterns_for_entry` IS called.
#
# Note: For testing `BaseStrategy` itself, a minimal concrete subclass might be needed
# to instantiate it, where abstract methods are implemented with simple return values
# or mocks. Alternatively, structure tests to call methods on `BaseStrategy.method(instance_of_subclass, ...)`.
# However, `pytest` typically allows direct testing of methods if the class can be instantiated,
# often by mocking out the abstract methods directly on the instance if they are not core to the
# specific unit test.
#
# Consider using hypothesis for property-based testing of `_prepare_data` with varying DataFrame inputs.
#
# Fixtures in `pytest` can be used extensively to set up mock objects (api, db, logger, config)
# and sample data (kline_data_raw, pattern_details dicts).
