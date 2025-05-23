import numpy as np
import pandas as pd
import MyTT as mt
import math

def calc_zlema(price, length):
    ema1 = mt.EMA(price, length)
    ema2 = mt.EMA(ema1, length)
    d = ema1 - ema2
    return ema1+d

def round_(val):
    return .999 if val > .99 else (-.999 if val < -.99 else val)

def IMACD(ohlc, lengthMA, lengthSignal):
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    tp = (high + low + close)/3
    
    hi = mt.MA(high, lengthMA)
    lo = mt.MA(low, lengthMA)
    mi = calc_zlema(tp, lengthMA)
    
    md = [x[0]-x[1] if x[0]>x[1] else (x[0]-x[2] if x[0]<x[2] else 0) for x in zip(mi, hi, lo)]
    sl = mt.MA(md, lengthSignal)
    sh = md - sl
    return md, sl, sh

def FISHER_V1(ohlc, length):
    med = (ohlc["high"] + ohlc["low"]) / 2
    ndaylow = med.rolling(window=length).min()
    ndayhigh = med.rolling(window=length).max()
    raw = 2 * ((med - ndaylow) / (ndayhigh - ndaylow)) - 1
    value = np.array([round_(x) for x in raw])
    fish = .5 * np.log((1 + value) / (1 - value))
    return fish

def FISHER_V2(ohlc, length, signal):
    hl2 = (ohlc["high"] + ohlc["low"]) / 2
    hl2_high = hl2.rolling(length).max()
    hl2_low = hl2.rolling(length).min()
    
    hlr = hl2_high - hl2_low
    hlr[hlr < 0.001] = 0.001
    
    raw = ((hl2 - hl2_low)/hlr) - 0.5
    
    v = 0
    m = hl2.size
    result = [np.nan for _ in range(0, length-1)] + [0]
    for i in range(length, m):
        v = 0.66 * raw.iloc[i] + 0.67 * v
        if v < -0.99: v = -0.999
        if v > 0.99: v = 0.999
        result.append(0.5 * (np.log((1+v)/(1-v)) + result[i-1]))
    fisher = pd.Series(result)
    fisherma = fisher.shift(1)
    return fisher, fisherma

def heikinashi(df):
    df_HA = df.copy()
    df_HA = df_HA.reset_index()
    df_HA['close']=(df_HA['open']+ df_HA['high']+ df_HA['low']+df_HA['close'])/4
    for i in range(0, len(df)):
        if i == 0:
            df_HA.loc[i,'open'] = (df_HA['open'][i] + df_HA['close'][i])/2
        else:
            df_HA.loc[i,'open'] = (df_HA['open'][i-1] + df_HA['close'][i-1])/2
    df_HA['high']=df_HA[['open','close','high']].max(axis=1)
    df_HA['low']=df_HA[['open','close','low']].min(axis=1)
    return df_HA[['close', 'open', 'high', 'low']]

def chandelier_exit(close, high, low, lookback=20, multi=3, use_close=False):
    atr1 = mt.ATR(close, high, low, 1)
    alpha = (1.0 / lookback) if lookback > 0 else 0.5
    atr = pd.Series(atr1).ewm(alpha=alpha, min_periods=lookback).mean()
    hh = high.rolling(lookback).max()
    ll = low.rolling(lookback).min()
    if use_close:
        hh = close.rolling(lookback).max()
        ll = close.rolling(lookback).min()

    long_stop = hh - multi * atr
    short_stop = ll + multi * atr

    _long_exit = long_stop.copy()
    _short_exit = short_stop.copy()
    _dir = [1] * len(long_stop)

    for i in range(len(close)):
        if i == 0:
            prev_long_stop = long_stop[i]
            prev_short_stop = short_stop[i]
            prev_close = close[i]
        else:
            prev_long_stop = _long_exit[i-1]
            prev_short_stop = _short_exit[i-1]
            prev_close = close[i-1]

        if math.isnan(prev_long_stop): prev_long_stop = long_stop[i]
        if math.isnan(prev_short_stop): prev_short_stop = short_stop[i]

        if prev_close > prev_long_stop:
            _long_exit[i] = max(prev_long_stop, long_stop[i])
        else:
            _long_exit[i] = long_stop[i]

        if prev_close < prev_short_stop:
            _short_exit[i] = min(prev_short_stop, short_stop[i])
        else:
            _short_exit[i] = short_stop[i]

        if close[i] > prev_short_stop: _dir[i] = 1
        elif close[i] < prev_long_stop: _dir[i] = -1
        else:
            _dir[i] = _dir[i-1]
    EXIT = [_long_exit[i] if _dir[i] > 0 else _short_exit[i] for i in range(len(_dir))]
    return EXIT, _dir

def cal_return(df, trend_type, multi):
    FEE_PERCENTAGE = 0.00025
    
    returns = []
    acc_return = []
    transaction_total = 0

    for index, row in df.iterrows():
        if index == len(df)-1:
            continue
        if index == 0:
            returns.append(0)
            acc_return.append(0)
            continue

        next_row = df.loc[index+1]
        last_row =  df.loc[index-1]

        transaction_fee = 0

        if row[trend_type] == 1:
            profit = (next_row['open'] - row['open']) * multi
        elif row[trend_type] == 0:
            profit = (row['open'] - next_row['open']) * multi
        else:
            profit = 0

        if row[trend_type] != last_row[trend_type]:
            transaction_fee = next_row['open'] * multi * FEE_PERCENTAGE
            
        profit_pct = (profit - transaction_fee) / (next_row['open']* multi)
        transaction_total += transaction_fee
        returns.append(profit_pct)
        acc_return.append(sum(returns))

    returns.append(0)
    acc_return.append(acc_return[-1])
    
    return returns, acc_return, transaction_total


def reversed_stop_profit(profit_list):
    profit1 = profit_list[-1]
    profit2 = max(profit_list[:-1])
    
    if profit2 > 0 and profit2 + stop_loss >= profit1:
        return True
    else:
        return False
    
def check_order_by_mode(order, current_price, mode, imacd_arr):
    # add IMACD stop method
    if mode == 'ATR':
        stop_profit = order['dynamic_stop_profit']
        stop_loss = order['dynamic_stop_loss']

        # 多倉
        if 'dynamic_long_price' in order:
            order_price = order['long_price']
            profit_list = order['profit_list']

            profit = current_price - order_price

            profit_list.append(profit / order_price)
            order['profit_list'] = profit_list

            if profit >= stop_profit or profit <= stop_loss:
                return True
    #         else:
    #             if len(profit_list) > 2:
    #                 if reversed_stop_profit(profit_list):
    #                     return True
            return False

        # 空倉
        elif 'dynamic_short_price' in order:
            order_price = order['short_price']
            profit_list = order['profit_list']

            profit = order_price - current_price

            profit_list.append(profit / order_price) 
            order['profit_list'] = profit_list

            if profit >= stop_profit or profit <= stop_loss:
                return True
    #         else:
    #             if len(profit_list) > 2:
    #                 if reversed_stop_profit(profit_list):
    #                     return True
            return False
    
    
def atrrsi(df, atr_column, short_period, long_period):
    atr_change = [0]
    atr_change_2 = [0]

    atr5rsi10 = []
    atr5rsi20 = []

    for i in range(long_period):
        if i < short_period:
            atr5rsi10.append(0)
        atr5rsi20.append(0)

    for index, row in df.iterrows():
        if index < 1:
            atr5rsi10.append(0) 
            atr5rsi20.append(0)
            continue

        prev_row = df.loc[index-1]

        if row[atr_column] - prev_row[atr_column] > 0:
            atr_change.append(row[atr_column] - prev_row[atr_column])
            atr_change_2.append(0)
        else:
            atr_change.append(0)
            atr_change_2.append(prev_row[atr_column] - row[atr_column])


        if index > short_period:
            change = sum(atr_change[len(atr_change)-short_period:]) / (sum(atr_change[len(atr_change)-short_period:]) + sum(atr_change_2[len(atr_change)-short_period:]))
            atr5rsi10.append(change)

        if index > long_period:
            change = sum(atr_change[len(atr_change)-long_period:]) / (sum(atr_change[len(atr_change)-long_period:]) + sum(atr_change_2[len(atr_change)-long_period:]))
            atr5rsi20.append(change) 
            
    return atr5rsi10, atr5rsi20


def rma(close, length):
    alpha = (1.0 / length) if length > 0 else 0.5
    rma = close.ewm(alpha=alpha, min_periods=length).mean()
    return rma


def rma_v2(src: pd.Series, length):
    src = src.fillna(0)
    alpha = 1 / length
    res = []
    for i in range(len(src)):
        if (i < length - 1):
            res.append(0)
        elif (i == length - 1):
            first = src[:i + 1]
            first = sum(first) / length
            res.append(first)
        else:
            resss = alpha * src[i] + (1 - alpha) * res[i - 1]
            res.append(round(resss, 3))
    return pd.Series(res)


def rsi(close, length):
    negative = close.diff()
    positive = negative.copy()

    positive[positive < 0] = 0  # Make negatives 0 for the postive series
    negative[negative > 0] = 0  # Make postives 0 for the negative series

    positive_avg = rma(positive, length=length)
    negative_avg = rma(negative, length=length)

    up = positive_avg
    down = negative_avg

    rsi = 100 * positive_avg / (positive_avg + negative_avg.abs())
    return rsi

def vol_heatmap(df):
    thresholdExtraHigh = 4
    thresholdHigh = 2.5
    thresholdMedium = 1
    thresholdNormal = -0.5

    df['hmean'] = df['volume'].rolling(610).mean()
    df['hstd'] = df['volume'].rolling(610).std()

    heatmap = []

    for index, row in df.iterrows():
        stdbar = (row.volume - row.hmean) / row.hstd
        conditionExtraHigh = (stdbar > thresholdExtraHigh)
        conditionHigh = (stdbar <= thresholdExtraHigh and stdbar > thresholdHigh)
        conditionMedium = (stdbar <= thresholdHigh and stdbar > thresholdMedium)
        conditionNormal = (stdbar <= thresholdMedium and stdbar > thresholdNormal)
        conditionLow = (stdbar <= thresholdNormal)
        dirt = (row.close > row.open)
        v = row.volume
        mosc = row.mean

        if conditionExtraHigh:
            heatmap.append(4)
        elif conditionHigh:
            heatmap.append(3)
        elif conditionMedium:
            heatmap.append(2)
        elif conditionNormal:
            heatmap.append(1)
        else:
            heatmap.append(0)
    return heatmap


def AMA(S, N=10, N1=2, N2=30):
    """
    :param S: 价格序列
    :param N: 价格区间
    :param N1: 快线
    :param N2: 慢线
    :return:
    """
    dif = abs(S[-1] - S[-N]) #价格变动量
    dif_sum = sum([abs(S[i-N] - S[i-N-1]) for i in range(N)[-N:]]) #价格波动值
    roc = dif / dif_sum #效率系数

    fast = 2 / (N1 + 1) # 快系数
    slow = 2 / (N2 + 1) # 慢系数
    s = roc * (fast - slow) + slow # 平滑系数
    c = s * s # 系数
    ama = EMA(DMA(S[-1], c), 2)

    return ama


def DMA(S, A):
    """
    :param S: 价格序列
    :param A: 平话因子，必须0<A<1, 公式为 Y=A*S+(1-A)*REF(Y,N)
    :return:
    """
    if isinstance(A, (int, float)):
        return pd.Series(S).ewm(alpha=A, adjust=False).mean().values

    A = np.array(A)
    A[np.isnan(A)] = 1.0
    Y = np.zeros(len(S))
    Y[0] = S[0]

    for i in range(1, len(S)):
        Y[i] = A[i] * S[i] + (1 - A[i]) * Y[i-1]
    return Y


def EMA(S, N):
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


length_bb = 9
mult_bb = 2
length_kc = 9
mult_kc = 2


def squeeze_momentum(df):
    # Calculate BB
    basis = df['close'].rolling(length_bb).mean()
    dev = mult_kc * df['close'].rolling(length_bb).std()
    upperBB = (basis + dev)
    lowerBB = (basis - dev)

    # Calculate KC
    ma = df['close'].rolling(length_kc).mean()
    atr = mt.ATR(df.close, df.high, df.low, length_kc)
    upperKC = (ma + atr * mult_kc)
    lowerKC = (ma - atr * mult_kc)

    signal = []

    for i in range(len(upperBB)):
        sqzOn = (lowerBB[i] > lowerKC[i]) and (upperBB[i] < upperKC[i])
        sqzOff = (lowerBB[i] < lowerKC[i]) and (upperBB[i] > upperKC[i])
        noSqz = (sqzOn == False) and (sqzOff == False)

        if not noSqz:
            if sqzOn:
                signal.append(0)
            else:
                signal.append(1)
        else:
            signal.append(2)

    return signal


def pullback_lines(df, length):
    df['hh'] = df['close'].rolling(length).max()
    df['ll'] = df['close'].rolling(length).min()
    df['dist'] = df['hh'] - df['ll']

    df['f0'] = np.where(df['close'] < df['close'].shift(length), df['hh'], df['ll'] + df['dist'])
    df['f1'] = np.where(df['close'] < df['close'].shift(length), df['hh'] - df['dist'] * 0.236,
                        df['ll'] + df['dist'] * 0.786)
    df['f2'] = np.where(df['close'] < df['close'].shift(length), df['hh'] - df['dist'] * 0.382,
                        df['ll'] + df['dist'] * 0.618)
    df['f3'] = np.where(df['close'] < df['close'].shift(length), df['hh'] - df['dist'] * 0.5,
                        df['ll'] + df['dist'] * 0.5)
    df['f4'] = np.where(df['close'] < df['close'].shift(length), df['hh'] - df['dist'] * 0.618,
                        df['ll'] + df['dist'] * 0.382)
    df['f5'] = np.where(df['close'] < df['close'].shift(length), df['hh'] - df['dist'] * 0.786,
                        df['ll'] + df['dist'] * 0.236)
    df['f6'] = np.where(df['close'] < df['close'].shift(length), df['hh'] - df['dist'], df['ll'])

    return df


def pivot_high(df, left, right=0):
    right = right if right else left
    df['pivot'] = 0
    for i in range(len(df)):
        if i >= left + right:
            rolling = df['high'][i - right - left:i + 1].values
            m = max(rolling)
            if df['high'][i - right] == m:
                df['pivot'].values[i] = m

    flag = 0
    pivot_high = []
    for _, row in df.iterrows():
        if row['pivot'] != 0:
            flag = row['pivot']
        pivot_high.append(flag)

    return pivot_high, df['pivot']


def pivot_low(df, left, right=0):
    right = right if right else left
    df['pivot'] = 0
    for i in range(len(df)):
        if i >= left + right:
            rolling = df['low'][i - right - left:i + 1].values
            m = min(rolling)
            if df['high'][i - right] == m:
                df['pivot'].values[i] = m

    flag = 0
    pivot_low = []
    for _, row in df.iterrows():
        if row['pivot'] != 0:
            flag = row['pivot']
        pivot_low.append(flag)

    return pivot_low, df['pivot']


def pivot_point_supertrend(high, low, close, pivot_length=2, atr_length=10, Factor=3):
    df = pd.DataFrame()
    df['high'] = high
    df['low'] = low
    df['close'] = close

    df['ph_line'], df['ph'] = pivot_high(df, pivot_length)
    df['pl_line'], df['pl'] = pivot_low(df, pivot_length)
    df['atr'] = mt.ATR(df.close, df.high, df.low, atr_length)

    # Initialize center as None
    center_list = []

    for _, row in df.iterrows():
        if len(center_list):
            last_center = center_list[-1]
        else:
            last_center = 0

        lastpp = row['ph'] if row['ph'] else (row['pl'] if row['pl'] else None)

        if lastpp:
            if not last_center:
                center = lastpp
            else:
                center = (last_center * 2 + lastpp) / 3
        else:
            center = last_center

        center_list.append(center)

    df['center'] = center_list

    # Calculate upper (Up) and lower (Dn) bands
    df['Up'] = df['center'] - Factor * df['atr']
    df['Down'] = df['center'] + Factor * df['atr']

    # Initialize TUp, TDown, Trend, and Trailingsl
    TUp = 0
    TDown = 0
    Trend = 0
    Trailingsl = None

    trends = []
    supertrends = []
    # Calculate TUp and TDown
    for index, row in df.iterrows():
        if index < 1:
            trends.append(0)
            supertrends.append(0)
            continue

        prev_row = df.loc[index - 1]

        Trend = 1 if row['close'] > TDown else (-1 if row['close'] < TUp else Trend)
        TUp = max(row['Up'], TUp) if prev_row['close'] > TUp else row['Up']
        TDown = min(row['Down'], TDown) if prev_row['close'] < TDown else row['Down']
        Trailingsl = TUp if Trend == 1 else TDown

        trends.append(Trend)
        supertrends.append(Trailingsl)

    return trends, supertrends

def isPivot(candle, window, data):
    """
    function that detects if a candle is a pivot/fractal point
    args: candle index, window before and after candle to test if pivot
    returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
    """
    if candle - window < 0 or candle + window >= len(data):
        return 0

    pivotHigh = 1
    pivotLow = 2
    for i in range(candle - window, candle + window + 1):
        if data.iloc[candle].low > data.iloc[i].low:
            pivotLow = 0
        if data.iloc[candle].high < data.iloc[i].high:
            pivotHigh = 0
    if (pivotHigh and pivotLow):
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0


def detect_macd_divergen(data):
    # 判断金叉
    data['JC'] = (data['DIFF'] > data['DEA']) & (data['DIFF'].shift(1) <= data['DEA'].shift(1))
    # 判断死叉
    data['SC'] = (data['DIFF'] < data['DEA']) & (data['DIFF'].shift(1) >= data['DEA'].shift(1))

    # 计算距离上次金叉的周期数
    data['N1'] = np.nan
    for i in range(len(data)):
        if data['JC'].iloc[i]:
            data.loc[i:, 'N1'] = 1
        elif pd.notna(data['N1'].iloc[i - 1]):
            data.loc[i, 'N1'] = data['N1'].iloc[i - 1] + 1

    # 计算距离上次死叉的周期数
    data['N2'] = np.nan
    for i in range(len(data)):
        if data['SC'].iloc[i]:
            data.loc[i:, 'N2'] = 1
        elif pd.notna(data['N2'].iloc[i - 1]):
            data.loc[i, 'N2'] = data['N2'].iloc[i - 1] + 1

    data['N1'] = data['N1'].fillna(0)
    data['N2'] = data['N2'].fillna(0)

    # 初始化列
    data['HH'] = np.nan
    data['HH2'] = np.nan
    data['MHD'] = np.nan
    data['MHD2'] = np.nan
    data['LL'] = np.nan
    data['LL2'] = np.nan
    data['MLD'] = np.nan
    data['MLD2'] = np.nan

    # 计算 HH 和 MHD
    last_HH = np.nan
    last_MHD = np.nan
    for i in range(len(data)):
        if data['SC'].iloc[i]:
            start_index = max(0, i - int(data['N1'].iloc[i]) + 1)
            data.loc[i, 'HH'] = data['high'].iloc[start_index:i + 1].max()
            data.loc[i, 'MHD'] = data['MACD'].iloc[start_index:i + 1].max()
            data.loc[i, 'HH2'] = last_HH
            data.loc[i, 'MHD2'] = last_MHD
            last_HH = data['HH'].iloc[i]
            last_MHD = data['MHD'].iloc[i]

    # 计算 LL 和 MLD
    last_LL = np.nan
    last_MLD = np.nan
    for i in range(len(data)):
        if data['JC'].iloc[i]:
            start_index = max(0, i - int(data['N2'].iloc[i]) + 1)
            data.loc[i, 'LL'] = data['low'].iloc[start_index:i + 1].min()
            data.loc[i, 'MLD'] = data['MACD'].iloc[start_index:i + 1].min()
            data.loc[i, 'LL2'] = last_LL
            data.loc[i, 'MLD2'] = last_MLD
            last_LL = data['LL'].iloc[i]
            last_MLD = data['MLD'].iloc[i]

    # 判断顶背离
    data['A'] = data['SC'] & (data['HH'] > data['HH2']) & (data['MHD'] < data['MHD2'])
    # 判断底背离
    data['B'] = data['JC'] & (data['LL'] < data['LL2']) & (data['MLD'] > data['MLD2'])

    return data


# 如果结构内的高低点不满足盈亏比，往前找
def check_prev_pivot_point(datetime, direction, entry_price, stop_loss, look_back_period, df):
    data_df = df.loc[df.datetime <= datetime].tail(int(look_back_period * 1.5))
    data_df = data_df.reset_index()
    data_df['is_pivot_v2'] = data_df.apply(lambda x: isPivot(x.name, 2, data_df), axis=1)

    pivot_datetime = None
    pivot_value = None
    localdf = data_df[-look_back_period:]

    if entry_price == stop_loss:
        return pivot_datetime, pivot_value

    # 找打最快能让仓位满足2倍盈亏比的前高
    if direction == 'LONG':
        for _, row in localdf.iterrows():
            if row['is_pivot_v2'] == 1:
                if (row['high'] - entry_price) / (entry_price - stop_loss) > 2:
                    if pivot_value:
                        if row['high'] < pivot_value:
                            pivot_value = row['high']
                            pivot_datetime = row['datetime']
                    else:
                        pivot_value = row['high']
                        pivot_datetime = row['datetime']

    elif direction == 'SHORT':
        for _, row in localdf.iterrows():
            if row['is_pivot_v2'] == 2:
                if (entry_price - row['low']) / (stop_loss - entry_price) > 2:
                    if pivot_value:
                        if row['low'] > pivot_value:
                            pivot_value = row['low']
                            pivot_datetime = row['datetime']
                    else:
                        pivot_value = row['low']
                        pivot_datetime = row['datetime']

    return pivot_datetime, pivot_value