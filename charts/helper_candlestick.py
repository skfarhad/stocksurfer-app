#!/usr/bin/env python

__author__ = "Sk Farhad"
__copyright__ = "Copyright (c) 2021 The Project owner- Sk Farhad (sk.farhad.eee@gmail.com)"

import os
import numpy as np
import pandas as pd
import mplfinance as mplf
import talib
from scipy import stats
import talib as ta
from pyti.bollinger_bands import upper_bollinger_band as bb_up
from pyti.bollinger_bands import middle_bollinger_band as bb_mid
from pyti.bollinger_bands import lower_bollinger_band as bb_low
HISTORY_FOLDER = 'dse_history_data'


def get_n_short(data_len):
    return data_len//4


def get_resampled(df, step='3D'):
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Trade': 'mean',
        'Volume': 'mean'
    }

    # resampled dataframe
    # 'W' means weekly aggregation
    df = df.resample(step).agg(agg_dict)
    return df


def process_data_mpl(file_path='', resample=False, step='3D'):
    df = pd.read_csv(
        file_path,
        sep=r'\s*,\s*',
        header=0,
        encoding='ascii',
        engine='python',
    )
    # print(df.head())
    columns = ['DATE', 'CLOSEP', 'VALUE_MN', 'HIGH', 'LOW', 'OPENP', 'TRADE']
    df = df[columns]
    df = df.rename(columns={
        'DATE': 'Date',
        'OPENP': 'Open',
        'CLOSEP': 'Close',
        'VALUE_MN': 'Volume',
        'TRADE': 'Trade',
        'HIGH': 'High',
        'LOW': 'Low'
    }, inplace=False)

    df = df[df['Open'] > 0]
    df = df[df['High'] > 0]
    df = df[df['Close'] > 0]

    df['Date'] = pd.to_datetime(df['Date'])
    if resample:
        df.index = pd.DatetimeIndex(df['Date'])
        df = get_resampled(df, step=step)
        df.reset_index(level=0, inplace=True)
        df.sort_values(by='Date', inplace=True)
        df.index = pd.DatetimeIndex(df['Date'])
        df.dropna(inplace=True)
    else:

        df.sort_values(by='Date', inplace=True)
        # df.reset_index(inplace=True)
        df.index = pd.DatetimeIndex(df['Date'])
    return df


def add_bb_plots(plots, data, period=20, panel=0):
    data_cl = data['Close'].values.tolist()
    bb_u = bb_up(data_cl, period)
    bb_l = bb_low(data_cl, period)
    bb_m = bb_mid(data_cl, period)

    bb_m_plot = mplf.make_addplot(bb_m, panel=panel, color='cyan', width=1, alpha=0.5)
    bb_l_plot = mplf.make_addplot(bb_l, panel=panel, color='yellow', width=1, alpha=0.3)
    bb_u_plot = mplf.make_addplot(bb_u, panel=panel, color='yellow', width=1, alpha=0.3)
    plots.extend([
        bb_u_plot, bb_m_plot, bb_l_plot
    ])


def add_macd_plots(plots, data, color_up, color_down, panel=1):
    macd, macd_signal, macd_hist = ta.MACD(
        data['Close'], fastperiod=10, slowperiod=22, signalperiod=7
    )

    colors = [color_up if v >= 0 else color_down for v in macd_hist]
    macd_plot = mplf.make_addplot(
        macd, panel=panel, color='orange', width=1, ylabel='MACD',
        secondary_y=False,
        y_on_right=False
    )
    macd_hist_plot = mplf.make_addplot(
        macd_hist, type='bar', panel=panel, color=colors,
        secondary_y=False

    )
    macd_signal_plot = mplf.make_addplot(
        macd_signal, panel=panel, color='blue', width=1,
        secondary_y=False
    )

    plots.extend([
        macd_plot, macd_signal_plot, macd_hist_plot
    ])


def add_rsi_plot(plots, data, color_up, color_down, panel=0, timeperiod=10):
    n_data = len(data)
    rsi = talib.RSI(data['Close'], timeperiod=timeperiod)

    line_rsi = mplf.make_addplot(
        rsi, panel=panel, color='gray', ylabel='RSI', width=1.5,
    )

    line_os = mplf.make_addplot(
        [70] * n_data, panel=panel,
        color=color_down, alpha=.5, linestyle='dashed', width=1.5,
        secondary_y=False
    )
    line_ob = mplf.make_addplot(
        [30] * n_data,
        panel=panel,
        color=color_up, alpha=.5, linestyle='dashed', width=1.5,
        secondary_y=False,
        ylabel='RSI'
    )

    plots.extend([
        line_os, line_rsi, line_ob
    ])


def add_line_plots(plots, data, panel=0):
    data_n = len(data)
    n_short = get_n_short(data_n)
    data_short = data['Close'][-n_short:]
    x_tr = range(0, len(data_short))
    slope, y_tr, r_val, p_val, std_err = stats.linregress(x_tr, data_short)
    y_tr_value = slope * x_tr + y_tr
    y_tr_value = np.concatenate((
        [np.NaN] * (data_n - n_short), y_tr_value
    ))

    data_long = data['Close'][: (data_n - n_short)]
    x_tr2 = range(0, len(data_long))
    slope2, y_tr2, r_val, p_val, std_err = stats.linregress(x_tr2, data_long)
    y_tr_value2 = slope2 * x_tr2 + y_tr2
    y_tr_value2 = np.concatenate((
        y_tr_value2, [np.NaN] * n_short
    ))

    y_tr_plot = mplf.make_addplot(
        y_tr_value, panel=panel,
        color='coral', width=2, alpha=0.4, linestyle='dashed'
    )
    price_plot = mplf.make_addplot(
        data['Close'].rolling(window=5).mean(),
        panel=panel,
        color='white', width=1, alpha=0.5
    )
    y_tr_plot2 = mplf.make_addplot(
        y_tr_value2, panel=panel,
        color='coral', width=2, alpha=0.4, linestyle='dashed'
    )

    plots.extend([
        y_tr_plot, price_plot, y_tr_plot2
    ])


def add_vol_plots(plots, data, color_up, color_down, vol_panel=2):
    vol_colors = data.apply(
        lambda x: color_up if x['Close'] > x['Open'] else color_down,
        axis=1
    )
    volume_plot = mplf.make_addplot(
        data['Volume'],
        panel=vol_panel,
        color=vol_colors.values,
        type='bar',
        ylabel='Value(Mil)',
    )
    plots.append(volume_plot)


def candelstick_plot(symbol, data, step='1D'):
    color_up = 'limegreen'
    color_down = 'tomato'
    plots = []

    add_rsi_plot(plots, data, color_up, color_down, panel=0)
    add_line_plots(plots, data, panel=1)
    add_bb_plots(plots, data, period=20, panel=1)

    add_macd_plots(plots, data, color_up, color_down, panel=2)
    add_vol_plots(plots, data, color_up, color_down, vol_panel=3)
    custom_nc = mplf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors={'candle': {'up': color_up, 'down': color_down},
                      'edge': {'up': color_up, 'down': color_down},
                      'wick': {'up': color_up, 'down': color_down},
                      'ohlc': {'up': color_up, 'down': color_down},
                      'volume': {'up': color_up, 'down': color_down},
                      'vcdopcod': True,  # Volume Color Depends On Price Change On Day
                      'alpha': 1.0,
                      },
        mavcolors=['gray', 'sienna', 'darkslategray', 'purple'],
    )
    data_mpl = data[['Date', 'Open', 'Close', 'Volume', 'High', 'Low']]
    fig, axlist = mplf.plot(
        data_mpl,
        type='candle',
        main_panel=1,
        style=custom_nc,
        title=symbol + ': ' + step,
        ylabel='Price (Tk)',
        figratio=(15, 7),
        addplot=plots,
        scale_padding={'left': 1, 'top': 1, 'right': 1, 'bottom': 1},
        panel_ratios=(0.3, 1, 0.3, .3),
        xrotation=7.5,
        tight_layout=True,
        # show_nontrading=True,
        returnfig=True
    )
    return fig


def get_candlestick_fig(
    symbol, data_n=120, resample=False, step='3D'
):
    try:
        path = os.path.join(HISTORY_FOLDER, symbol + '_history_data.csv')
        data = process_data_mpl(path, resample=resample, step=step)
        data = data[-data_n:]
        if not resample:
            step = '1D'
        fig = candelstick_plot(symbol, data, step=step)
        return fig
    except Exception as e:
        print(str(e))
        return None




