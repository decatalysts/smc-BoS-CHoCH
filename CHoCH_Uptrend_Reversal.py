from smartmoneyconcepts import smc
from tqsdk import TqApi, TqAuth, tafunc
from MyTT import MACD
from utils import isPivot, detect_macd_divergen, check_prev_pivot_point

# 交易参数
max_loss_per_trade = 1000
wait_windows = 30


def find_entry(choch_up, product_info, df, row, current_datetime, product_multiplier, logging):
    print('UPTREND REVERSAL', choch_up, product_info['product'], df.index[-1])
    logging.info(f'UPTREND REVERSAL, {choch_up}, {product_info["product"]}, {df.index[-1]}')
    reverse_index = df.loc[df.datetime == choch_up['reverse_datetime']].index.values[0]
    if df.index[-1] - reverse_index > wait_windows or df.index[-1] == reverse_index:
        return
    else:
        is_macd_divergence = False
        for entry in choch_up['entries']:
            # FVG的失衡区时，K线必须在FVG的后两根
            if entry['type'] == 'fvg':
                fvg_index = df.loc[df.datetime == entry['datetime']].index.values[0]
                if row['index'] - fvg_index < 2:
                    return

            stop_loss = entry['upper_range']
            after_break_df = df.loc[
                (df.datetime >= choch_up['reverse_datetime']) & (df.datetime <= row['datetime'])]
            if stop_loss < max(after_break_df['high'].values):
                stop_loss = max(after_break_df['high'].values)

            take_profit = min(after_break_df['low'].values)
            min_index = after_break_df['low'].idxmin()
            take_profit_datetime = df.loc[min_index].datetime

            is_reversed = choch_up['prev_high'] < max(after_break_df['close'].values) or choch_up[
                'prev_high'] < max(after_break_df['open'].values)

            pattern_df = df.loc[
                (df.datetime >= choch_up['pattern_start_datetime']) & (df.datetime <= row['datetime'])]
            if len(pattern_df.loc[pattern_df.B]):
                is_macd_divergence = True
                if df.loc[min_index].close > df.loc[min_index].open:
                    take_profit = df.loc[min_index].open
                else:
                    take_profit = df.loc[min_index].close

            # 计算盈亏比是否在2以上
            profit_factor = 0
            if row['close'] != stop_loss:
                profit_factor = round((take_profit - row['close']) / (row['close'] - stop_loss), 1)

            if profit_factor < 2 and not is_macd_divergence:
                pivot_datetime, pivot_value = check_prev_pivot_point(row['datetime'], 'SHORT',
                                                                     row['close'], stop_loss, 300, df)

                if pivot_value and row['close'] != stop_loss:
                    take_profit = pivot_value
                    take_profit_datetime = pivot_datetime
                    profit_factor = round((take_profit - row['close']) / (row['close'] - stop_loss), 1)
            print(product_info['product'], is_reversed)
            logging.info(f"{product_info['product']}, {is_reversed}, choch up, {choch_up['reverse_datetime']}"
                         f"{row['close']}, {row['open']}, {stop_loss}")

            if is_reversed:
                logging.info(f"is_reversed, {product_info['product']}, {is_reversed}")

            if not is_reversed:
                if row['close'] == stop_loss:
                    return

                lots = round(max_loss_per_trade / (abs(row['close'] - stop_loss) * product_multiplier))
                if entry['lower_range'] <= row['high'] <= entry['upper_range'] and row['close'] < row['open']:
                    print('SHORT', product_info['product'], current_datetime, choch_up['reverse_datetime'],
                          entry['type'], stop_loss, 'high', profit_factor,
                          take_profit, take_profit_datetime, is_macd_divergence, row['datetime'], lots)
                    logging.warning(f"SHORT, {product_info['product']}, {current_datetime},"
                                 f" {choch_up['reverse_datetime']}, {entry['type']}, {stop_loss}, high,"
                                 f"{profit_factor}, {take_profit}, "
                                 f"{is_macd_divergence}, {row['datetime']}, {lots}")

                if entry['lower_range'] <= row['open'] <= entry['upper_range'] and row['close'] < row['open']:
                    print('SHORT', product_info['product'], current_datetime, choch_up['reverse_datetime'],
                          entry['type'], stop_loss, 'open', profit_factor,
                          take_profit, take_profit_datetime, is_macd_divergence, row['datetime'], lots)
                    logging.warning(f"SHORT, {product_info['product']}, {current_datetime},"
                                 f" {choch_up['reverse_datetime']}, {entry['type']}, {stop_loss}, open,"
                                 f"{profit_factor}, {take_profit}, "
                                 f"{is_macd_divergence}, {row['datetime']}, {lots}")

                if entry['lower_range'] <= row['low'] <= entry['upper_range'] and row['close'] < row['open']:
                    print('SHORT', product_info['product'], current_datetime, choch_up['reverse_datetime'],
                          entry['type'], stop_loss, 'low', profit_factor,
                          take_profit, take_profit_datetime, is_macd_divergence, row['datetime'], lots)
                    logging.warning(f"SHORT, {product_info['product']}, {current_datetime},"
                                 f" {choch_up['reverse_datetime']}, {entry['type']}, {stop_loss}, low,"
                                 f"{profit_factor}, {take_profit}, "
                                 f"{is_macd_divergence}, {row['datetime']}, {lots}")

                if entry['lower_range'] <= row['close'] <= entry['upper_range'] and row['close'] < row['open']:
                    print('SHORT', product_info['product'], current_datetime, choch_up['reverse_datetime'],
                          entry['type'], stop_loss, 'close', profit_factor,
                          take_profit, take_profit_datetime, is_macd_divergence, row['datetime'], lots)
                    logging.warning(f"SHORT, {product_info['product']}, {current_datetime},"
                                    f" {choch_up['reverse_datetime']}, {entry['type']}, {stop_loss}, close,"
                                    f"{profit_factor}, {take_profit}, "
                                    f"{is_macd_divergence}, {row['datetime']}, {lots}")


def detect_CHoCH_Uptrend_Reversal(api, logging, product_info, db):
    kline_data = product_info['kline_data']
    quote = product_info['quote']
    product_multiplier = quote.volume_multiple

    if api.is_changing(kline_data.iloc[-1], "datetime"):
        current_datetime = tafunc.time_to_datetime(kline_data.iloc[-1]['datetime']).strftime('%Y-%m-%d %H:%M:%S')
        df = kline_data.copy()
        df = df.iloc[:-1]
        df = df.reset_index()
        datetimes = []
        for _, row in df.iterrows():
            datetimes.append(tafunc.time_to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M:%S'))
        df['datetime'] = datetimes

        df['DIFF'], df['DEA'], df['MACD'] = MACD(df['close'])
        df = detect_macd_divergen(df)
        window = 5
        df['is_pivot'] = df.apply(lambda x: isPivot(x.name, window, df), axis=1)
        row = df.iloc[-1]

        localdf = df[-41:-1]
        pivot_df = localdf.loc[localdf.is_pivot != 0][['datetime', 'is_pivot', 'high', 'low']]
        pivot_df['origin_index'] = pivot_df.index
        pivot_df = pivot_df.reset_index()
        pivot_df['group'] = (pivot_df['is_pivot'] != pivot_df['is_pivot'].shift()).cumsum()
        result = pivot_df.groupby('group').last()

        highs = result[result['is_pivot'] == 1].high.tail(2).values
        idxhighs = result[result['is_pivot'] == 1].origin_index.tail(2).values
        lows = result[result['is_pivot'] == 2].low.tail(2).values
        idxlows = result[result['is_pivot'] == 2].origin_index.tail(2).values

        choch_up = None

        if len(highs) == 2 and len(lows) == 2:
            order_condition = idxlows[0] < idxhighs[0] < idxlows[1] < idxhighs[1]

            pattern_1 = lows[0] < highs[0] < highs[1] and lows[0] < lows[1] < highs[0]

            if order_condition and pattern_1:
                data_df_2 = df.loc[idxhighs[1] + 1:]

                reverse_index = 0
                for idx, sub_row in data_df_2.iterrows():
                    if sub_row['low'] < lows[1]:
                        reverse_index = idx
                        break

                if reverse_index != 0:
                    swing_highs_lows = smc.swing_highs_lows(df, swing_length=4)
                    ob_data = smc.ob(df, swing_highs_lows, close_mitigation=False)

                    ob_idxs = []
                    fvg_idxs = []

                    for index, sub_row in ob_data.iterrows():
                        if sub_row['OB'] == -1:
                            ob_idxs.append(index)

                    fvg_data = smc.fvg(df, join_consecutive=False)
                    for index, sub_row in fvg_data.iterrows():
                        if sub_row['FVG'] == -1:
                            fvg_idxs.append(index)

                    # 找到合适的OB
                    obs = [x for x in ob_idxs if idxhighs[1] <= x <= reverse_index]
                    obs_entry = []
                    if len(obs):
                        obs_entry.append({
                            'upper_range': df.loc[obs[-1]].high,
                            'lower_range': df.loc[obs[-1]].low,
                            'datetime': df.loc[obs[-1]].datetime,
                            'type': 'ob',
                        })

                    # 找到合适的FVG
                    fvgs = [x for x in fvg_idxs if idxhighs[1] <= x <= reverse_index + 2]
                    fvgs_entry = []
                    if len(fvgs):
                        for idx in fvgs:
                                fvgs_entry.append({
                                    'upper_range': fvg_data.loc[idx].Top,
                                    'lower_range': fvg_data.loc[idx].Bottom,
                                    'datetime': df.loc[idx].datetime,
                                    'type': 'fvg',
                                })

                    # 没有OB，用high1来当OB
                    if not len(obs):
                        obs_entry.append({
                            'upper_range': df.loc[idxhighs[1]].high,
                            'lower_range': df.loc[idxhighs[1]].low,
                            'datetime': df.loc[idxhighs[1]].datetime,
                            'type': 'high1',
                        })

                    if len(fvgs_entry) or len(obs_entry):
                        choch_up = {
                            'entries': fvgs_entry + obs_entry,
                            'reverse_datetime': df.loc[reverse_index].datetime,
                            'reverse_index': reverse_index,
                            'prev_high': highs[1],
                            'pattern_start_datetime': df.loc[idxlows[0]].datetime,
                            'pivot_points': [df.loc[idxlows[0]].datetime, df.loc[idxhighs[0]].datetime,
                                             df.loc[idxlows[1]].datetime, df.loc[idxhighs[1]].datetime],
                            'pivot_values': [lows[0], highs[0], lows[1], highs[1]],
                            'product': product_info['product']
                        }

        # 当前时间跟突破K相距不超过30个K线
        if choch_up:
            find_entry(choch_up, product_info, df, row, current_datetime, product_multiplier, logging)

            hist_choch_up = db.find_by_condition('choch_uptrend', {
                'product': product_info['product'],
                'reverse_datetime': choch_up['reverse_datetime']
            })
            if not hist_choch_up:
                choch_up['detect_datetime'] = current_datetime
                db.insert('choch_uptrend', choch_up)
        else:
            # 读取存储的结构，但只用window内的数据
            look_back_datetime = df.iloc[-wait_windows].datetime
            choch_ups = db.find_by_condition('choch_uptrend', {
                'product': product_info['product'],
                'reverse_datetime': {'$gte': look_back_datetime}
            })

            if choch_ups:
                for choch_up in choch_ups:
                    find_entry(choch_up, product_info, df, row, current_datetime, product_multiplier, logging)
                if len(choch_ups) == 0:
                    logging.info(f"{product_info['product']}, CHoCH Uptrend, look_back_datetime: {look_back_datetime}")
            else:
                logging.info(f"{product_info['product']}, CHoCH Uptrend, look_back_datetime: {look_back_datetime}")