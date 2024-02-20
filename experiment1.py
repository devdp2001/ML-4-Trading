import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data
import marketsimcode as ms
import ManualStrategy as ml
import StrategyLearner as sl

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "dpatel426"  # Change this to your user ID

def runexp1():
    # In-Sample
    symbol = 'JPM'
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')

    # Benchmark portval
    benchmarkJPM = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')
    benchmarkJPM[:] = 0
    benchmarkJPM.iloc[0, 0] = 1000
    benchmark_portval = ms.compute_portvals(symbol=symbol, orders_df=benchmarkJPM, start_val=sv, commission=9.95, impact=0.005)

    # Manual Strategy Portval
    manlearner = ml.ManualStrategy()
    manual_trades = manlearner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    manual_portval = ms.compute_portvals(symbol=symbol, orders_df=manual_trades, start_val=sv, commission=9.95, impact=0.005)

    #Strategy Learner Portval
    stratlearner = sl.StrategyLearner(verbose = False, impact = 0.005, commission=9.95)
    stratlearner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strat_trades = stratlearner.testPolicy(symbol=symbol, sd=sd, ed=ed,sv=sv)
    strat_portval = ms.compute_portvals(symbol=symbol, orders_df=strat_trades, start_val=sv, commission=9.95,impact=0.005)

    # Normalize
    benchmark_portval = benchmark_portval / benchmark_portval[0]
    manual_portval = manual_portval / manual_portval[0]
    strat_portval = strat_portval / strat_portval[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(benchmark_portval.index, benchmark_portval.values, color='purple', label='Benchmark')
    ax.plot(manual_portval.index, manual_portval.values, color='red', label='Manual')
    ax.plot(strat_portval.index, strat_portval.values, color='green', label='Strategy Learner')
    plt.title('Benchmark vs. Manual vs. Strategy Portfolio Value for JPM In-Sample (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.legend(loc='upper left')
    plt.grid(visible=True)
    plt.savefig('images/ManualVsBenchVsStrategyInSample.png')
    plt.clf()

    # Out-Sample
    symbol = 'JPM'
    sv = 100000
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')

    # Benchmark portval
    benchmarkJPM = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')
    benchmarkJPM[:] = 0
    benchmarkJPM.iloc[0, 0] = 1000
    benchmark_portval = ms.compute_portvals(symbol=symbol, orders_df=benchmarkJPM, start_val=sv, commission=9.95, impact=0.005)

    # Manual Strategy Portval
    manlearner2 = ml.ManualStrategy()
    manual_trades = manlearner2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    manual_portval = ms.compute_portvals(symbol=symbol, orders_df=manual_trades, start_val=sv, commission=9.95, impact=0.005)

    #Strategy Learner Portval
    stratlearner2 = sl.StrategyLearner(verbose = False, impact = 0.005, commission=9.95)
    stratlearner2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strat_trades = stratlearner2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strat_portval = ms.compute_portvals(symbol=symbol, orders_df=strat_trades, start_val=sv, commission=9.95, impact=0.005)

    # Normalize
    benchmark_portval = benchmark_portval / benchmark_portval[0]
    manual_portval = manual_portval / manual_portval[0]
    strat_portval = strat_portval / strat_portval[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(benchmark_portval.index, benchmark_portval.values, color='purple', label='Benchmark')
    ax.plot(manual_portval.index, manual_portval.values, color='red', label='Manual')
    ax.plot(strat_portval.index, strat_portval.values, color='green', label='Strategy Learner')
    plt.title('Benchmark vs. Manual vs. Strategy Portfolio Value for JPM Out-Sample (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.legend(loc='upper left')
    plt.grid(visible=True)
    plt.savefig('images/ManualVsBenchVsStrategyOutSample.png')
    plt.clf()
