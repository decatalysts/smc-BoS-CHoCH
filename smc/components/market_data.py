from tqsdk import TqApi, TqSim, TqChan, tafunc
import asyncio
from typing import Dict, Tuple, Any
from datetime import datetime

class MarketData:
    def __init__(self, api: TqApi):
        self.api = api
        self.quotes: dict[str, any] = {}
        # 1. 管理 Quote 订阅对应的 TqChan
        #    key: symbol，例如 "SHFE.rb2510"
        #    value: TqChan 对象
        self.quote_channels: Dict[str, TqChan] = {}
        # 2. 管理 K 线订阅对应的 (kline_obj, TqChan)
        #    key: "symbol_tf"，例如 "SHFE.rb2510_15m", "SHFE.rb2510_30m"
        #    value: (kline_obj, TqChan)
        self.kline_channels: Dict[str, Tuple[Any, TqChan]] = {}

    def _interval_to_seconds(self, interval: str) -> int:
        """将时间间隔转换为秒数"""
        multiplier = int(interval[:-1])
        unit = interval[-1]

        if unit == 'm':
            return multiplier * 60
        elif unit == 'h':
            return multiplier * 3600
        elif unit == 'd':
            return multiplier * 86400
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    def subscribe_quote(self, symbol: str) -> TqChan:
        """
        订阅一个合约的实时 Quote 数据（Tick）。
        每当该合约的最新价格、盘一价或成交量变化时，就把最新的 Tick 打包推送给返回的 TqChan。
        """
        if symbol not in self.quote_channels:
            # 1. 在 TqSdk 中获取 Quote 对象引用
            self.quotes[symbol] = self.api.get_quote(symbol)
            # 2. 创建一个 TqChan（last_only=True：只保留最新一条数据）
            chan = TqChan(self.api, last_only=True)
            self.quote_channels[symbol] = chan
            # 我们把 quote_obj 暂存到 TqChan 里（或可存到另一个字典）
            # 但可以通过遍历 quote_channels 的 key 来拿到 quote_obj：
            #   quote_obj = self.api.get_quote(symbol)
            # 当 run() 中检测 is_changing 时，直接调用 get_quote 再推送
        return self.quote_channels[symbol]

    def subscribe_kline(self, symbol: str, tf: str, kline_length: int = 500) -> TqChan:
        """
        订阅一个合约的实时 K 线数据，tf 为时间周期字符串，如 "15m", "30m", "1h"。
        每当这根 tf 周期的 K 线闭合时，就把该 Bar 数据打包推送给对应的 TqChan。
        """
        duration_seconds = self._interval_to_seconds(tf)

        key = f"{symbol}_{tf}"
        if key not in self.kline_channels:
            # 1. 直接调用 TqSdk 接口获取该周期的 K 线对象
            kline_obj = self.api.get_kline_serial(
                symbol=self.quotes[symbol].underlying_symbol,
                duration_seconds=duration_seconds,
                data_length=kline_length,
            )
            # 2. 创建对应的 TqChan
            chan = TqChan(self.api, last_only=True)
            self.kline_channels[key] = (kline_obj, chan)
        return self.kline_channels[key][1]

    async def run(self):
        """
        核心协程：无限循环监听 TqSdk 数据更新，并将变化推送到各个订阅的通道。
        """
        async with self.api.register_update_notify() as update_chan:
            async for _ in update_chan:
                # 1. 遍历每个已订阅的 Quote(合约)，检测是否发生变化
                for symbol, chan in self.quote_channels.items():
                    quote_obj = self.api.get_quote(symbol)  # 拿到对应的 Quote 对象
                    if self.api.is_changing(quote_obj):
                        # 将最新 Tick 数据打包成字典
                        data = {
                            "symbol":      quote_obj.underlying_symbol,
                            "datetime":    tafunc.time_to_datetime(quote_obj.datetime).strftime('%Y-%m-%d %H:%M:%S'),
                            "last_price":  quote_obj.last_price,
                            "bid_price1":  quote_obj.bid_price1,
                            "ask_price1":  quote_obj.ask_price1,
                            "volume":      quote_obj.volume,
                            "open_interest": getattr(quote_obj, "open_interest", None)
                        }
                        # 非阻塞推送到对应通道
                        await chan.send(data)

                # 2. 遍历每个已订阅的 K 线 (symbol_tf)
                for key, (kline_obj, chan) in self.kline_channels.items():
                    if self.api.is_changing(kline_obj.iloc[-1], "datetime"):
                        # key 里的格式为 "symbol_tf"；我们可以拆分出它们
                        symbol, tf = key.split("_", 1)
                        quote_obj = self.api.get_quote(symbol)
                        # 把最新完整闭合 K 线打包成字典
                        bar = {
                            "symbol":        quote_obj.underlying_symbol,
                            "tf":            kline_obj.duration,  # e.g., "15m"
                            "datetime":      tafunc.time_to_datetime(quote_obj.datetime).strftime('%Y-%m-%d %H:%M:%S'),  # K 线结束时间
                            "open":          kline_obj.open,
                            "high":          kline_obj.high,
                            "low":           kline_obj.low,
                            "close":         kline_obj.close,
                            "volume":        kline_obj.volume,
                            "open_interest": getattr(kline_obj, "open_interest", None)
                        }
                        await chan.send(bar)
