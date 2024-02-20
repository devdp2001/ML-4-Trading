import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "dpatel426"  # Change this to your user ID

def Simple_moving_Average(df, window):
    sma = df.rolling(window).mean()
    return sma

def Momentum(df, window):
    momentum = (df / df.shift(window)) - 1
    return momentum

def Moving_Average_Convergence_Divergence(df, s_window, l_window, signal):
    short_ema = df.ewm(span=s_window, ignore_na=False ,adjust=False).mean()
    long_ema = df.ewm(span=l_window, ignore_na=False ,adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal, ignore_na=False , adjust=False).mean()
    return macd_line, signal_line

def Bollinger_Bands_Percentage(df, window):
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    uband = mean + (std * 2)
    lband = mean - (std * 2)
    bbp = (df - lband)/(uband - lband)
    return bbp

def Commodity_Channel_Index(df, window):
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    cci = (df - mean) / ((2 * df.std()))
    return cci

def run():
    values = get_data(['JPM'], pd.date_range((dt.datetime(2008, 1,1)),(dt.datetime(2009, 12, 31))), addSPY=False).dropna(how='all')
    values = values/values.iloc[0]

    sma = Simple_moving_Average(values, 10)
    momentum = Momentum(values, 10)
    macd, signal = Moving_Average_Convergence_Divergence(values, 12, 26, 9)
    bb_ub, bb_lb = Bollinger_Bands(values, 10)
    cci = Commodity_Channel_Index(values, 10)

    #SMA
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(values.index, values)
    ax.plot(values.index, sma)
    plt.title('SMA Indicator for JPM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(['Price', 'SMA'], loc = 'lower right')
    plt.grid(visible = True)
    plt.savefig('SMA.png')
    plt.clf()

    #Momentum
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(values.index, momentum)
    ax.axhline(y=0, color='red', linestyle='--')
    plt.title('Momentum Indicator for JPM')
    plt.xlabel('Date')
    plt.ylabel('Momentum')
    plt.legend(['Momentum'], loc = 'lower right')
    plt.grid(visible = True)
    plt.savefig('Momentum.png')
    plt.clf()

    #MACD
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(values.index, macd)
    ax.plot(values.index, signal, linestyle='--', color='red')
    plt.title('MACD Indicator for JPM')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend(['MACD Line', 'Signal Line'], loc = 'lower right')
    plt.grid(visible = True)
    plt.savefig('MACD.png')
    plt.clf()

    #BB
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(values.index, values)
    ax.plot(values.index, bb_ub)
    ax.plot(values.index, bb_lb)
    plt.title('BB Indicator for JPM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(['Price', 'Upper Band', 'Lower Band'], loc = 'lower right')
    plt.grid(visible = True)
    plt.savefig('BB.png')
    plt.clf()
    #BBP
    bbp = (values - bb_lb)/(bb_ub-bb_lb)
    fig, ax = plt.subplots(figsize = (25, 10))
    ax.plot(values.index, bbp)
    ax.axhline(y=1, color='red', linestyle='--')
    ax.axhline(y=0, color='red', linestyle='--')
    plt.title('BB Percentage Indicator for JPM')
    plt.xlabel('Date')
    plt.ylabel('BB Percentage (%)')
    plt.legend(['BBP'], loc = 'lower right')
    plt.grid(visible = True)
    plt.savefig('BBP.png')
    plt.clf()

    #CCI
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.plot(values.index, cci)
    ax.axhline(y=.25, color='red', linestyle='--')
    ax.axhline(y=-.25, color='red', linestyle='--')
    plt.title('CCI Indicator for JPM')
    plt.xlabel('Date')
    plt.ylabel('CCI Value')
    plt.legend(['CCI'], loc = 'lower right')
    plt.grid(visible = True)
    plt.savefig('CCI.png')
    plt.clf()

