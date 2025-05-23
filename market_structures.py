# -*- coding: UTF-8 -*-
from tqsdk import TqApi, TqAuth, tafunc
import logging
from datetime import datetime
from BoS_Uptrend import detect_BoS_Uptrend
from BoS_Downtrend import detect_BoS_Downtrend
from CHoCH_Uptrend_Reversal import detect_CHoCH_Uptrend_Reversal
from CHoCH_Downtrend_Reversal import detect_CHoCH_Downtrend_Reversal
from db.db_controller import DBController

# LOGGING SETTING
log_file_name = datetime.today().strftime('%Y-%m-%d')
logging.basicConfig(filename=f'logs/{log_file_name}',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
# MongoDB Connection
DB_URL = "mongodb://localhost:27017/"
DB_NAME = "smc"
db = DBController(DB_URL, DB_NAME)

"""
    交易所/品种
"""
INE = ['sc', 'lu', 'nr', 'ec']
DCE = ['jm', 'j', 'i', 'm', 'eg', 'eb', 'v', 'y', 'p',
       'a', 'b', 'c', 'cs', 'jd', 'lh', 'l', 'pp', 'pg', 'lg']
CZCE = ['fg', 'sa', 'sh', 'ma', 'cf', 'ta', 'ur', 'pf', 'sr',
        'cy', 'oi', 'rm', 'pk', 'sf', 'sm', 'ap', 'cj', 'px', 'pr']
SHFE = ['ag', 'ru', 'rb', 'cu', 'al', 'zn', 'pb', 'ni',
        'sn', 'au', 'hc', 'ss', 'fu', 'bu', 'sp', 'ao', 'br']
GFEX = ['lc', 'si', 'ps']
CFFEX = ['IF', 'IC', 'IM']

api = TqApi(
    auth=TqAuth("18822034094", "liuleren123")
)

product_info = {}
freq = 15
kline_length = 500

for product in INE:
    quote = api.get_quote(f"KQ.m@INE.{product}")
    underlying_symbol = quote.underlying_symbol
    product_info[product] = {
        'quote': quote,
        'underlying_symbol': underlying_symbol,
        'kline_data': api.get_kline_serial(underlying_symbol, 60 * freq, kline_length),
        'product': product,
    }

for product in DCE:
    quote = api.get_quote(f"KQ.m@DCE.{product}")
    underlying_symbol = quote.underlying_symbol
    product_info[product] = {
        'quote': quote,
        'underlying_symbol': underlying_symbol,
        'kline_data': api.get_kline_serial(underlying_symbol, 60 * freq, kline_length),
        'product': product,
    }

for product in CZCE:
    product = product.upper()
    quote = api.get_quote(f"KQ.m@CZCE.{product}")
    underlying_symbol = quote.underlying_symbol
    product_info[product] = {
        'quote': quote,
        'underlying_symbol': underlying_symbol,
        'kline_data': api.get_kline_serial(underlying_symbol, 60 * freq, kline_length),
        'product': product,
    }

for product in SHFE:
    quote = api.get_quote(f"KQ.m@SHFE.{product}")
    underlying_symbol = quote.underlying_symbol
    product_info[product] = {
        'quote': quote,
        'underlying_symbol': underlying_symbol,
        'kline_data': api.get_kline_serial(underlying_symbol, 60 * freq, kline_length),
        'product': product,
    }

for product in GFEX:
    quote = api.get_quote(f"KQ.m@GFEX.{product}")
    underlying_symbol = quote.underlying_symbol
    product_info[product] = {
        'quote': quote,
        'underlying_symbol': underlying_symbol,
        'kline_data': api.get_kline_serial(underlying_symbol, 60 * freq, kline_length),
        'product': product,
    }
for product in CFFEX:
    quote = api.get_quote(f"KQ.m@CFFEX.{product}")
    underlying_symbol = quote.underlying_symbol
    product_info[product] = {
        'quote': quote,
        'underlying_symbol': underlying_symbol,
        'kline_data': api.get_kline_serial(underlying_symbol, 60 * freq, kline_length),
        'product': product,
    }


while True:
    api.wait_update()

    for product in product_info:
        detect_BoS_Uptrend(api, logging, product_info[product], db)
        detect_BoS_Downtrend(api, logging, product_info[product], db)
        detect_CHoCH_Downtrend_Reversal(api, logging, product_info[product], db)
        detect_CHoCH_Uptrend_Reversal(api, logging, product_info[product], db)