import os

from datetime import  date
from tqsdk import TqApi, TqAuth, TqBacktest
from smc.components.market_data import MarketData
from dotenv import load_dotenv
from contextlib import closing

def main():
    load_dotenv()

    # 初始化API
    api = TqApi(
        backtest=TqBacktest(
            start_dt=date(2025, 5, 14),
            end_dt=date(2025, 5, 15),
        ),
        auth=TqAuth(
            os.getenv('TQ_ACC_USERNAME'),
            os.getenv('TQ_ACC_PASSWORD'),
        ),
    )

    md = MarketData(api)
    # 订阅 RB主连合约 的实时 Quote 和 15m/30m K 线
    quote_chan = md.subscribe_quote("KQ.m@SHFE.rb")
    k15_chan   = md.subscribe_kline("KQ.m@SHFE.rb", "15m")
    k30_chan   = md.subscribe_kline("KQ.m@SHFE.rb", "30m")

    api.create_task(md.run())

    # 1. 消费 Tick：实时打印
    # async def print_ticks():
    #     while True:
    #         t = await quote_chan.recv()
    #         print(f"[TICK] {t['symbol']} {t['datetime']} 价={t['last_price']} 买1={t['bid_price1']} 卖1={t['ask_price1']} 成交量={t['volume']}")

    # 示例消费协程：每根15m Bar 闭合时打印
    async def print_k15():
        while True:
            bar = await k15_chan.recv()
            # print(f"[15m] {bar['symbol']} {bar['datetime']} O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']} V={bar['volume']}")
            print(f"[15m] {bar['symbol']} {bar['datetime']}")

    # 示例消费协程：每根30m Bar 闭合时打印
    # async def print_k30():
    #     while True:
    #         bar = await k30_chan.recv()
    #         print(f"[30m] {bar['symbol']} {bar['datetime']} O={bar['open']} H={bar['high']} L={bar['low']} C={bar['close']} V={bar['volume']}")

    # 注册到 TqApi 的任务队列
    # api.create_task(print_tick())
    api.create_task(print_k15())
    # api.create_task(print_k30())

    print("MarketData 运行中，按 Ctrl+C 退出")
    with closing(api):
        while True:
            api.wait_update()

if __name__ == '__main__':
    main()
