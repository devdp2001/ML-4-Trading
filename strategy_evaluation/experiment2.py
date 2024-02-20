import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data
import marketsimcode as ms
import StrategyLearner as sl


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "dpatel426"  # Change this to your user ID

def get_stats(port_val):
    daily_rets = (port_val / port_val.shift(1) - 1)
    cr = (port_val[-1] / port_val[0]) - 1
    sddr = daily_rets.std()
    adr = daily_rets.mean()
    return cr, sddr, adr

def runexp2():
    # In-Sample
    symbol = 'JPM'
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')


    #Strategy Learner Portval impact .05
    stratlearner = sl.StrategyLearner(verbose = False, impact = 0.05, commission=0)
    stratlearner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strat_trades = stratlearner.testPolicy(symbol=symbol, sd=sd, ed=ed,sv=sv)
    strat_portval = ms.compute_portvals(symbol=symbol, orders_df=strat_trades, start_val=sv, commission=0,impact=0.05)

    #Strategy Learner Portval impact .005
    stratlearner2 = sl.StrategyLearner(verbose = False, impact = 0.005, commission=0)
    stratlearner2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strat_trades2 = stratlearner.testPolicy(symbol=symbol, sd=sd, ed=ed,sv=sv)
    strat_portval2 = ms.compute_portvals(symbol=symbol, orders_df=strat_trades2, start_val=sv, commission=0,impact=0.005)

    #Strategy Learner Portval impact .0005
    stratlearner3 = sl.StrategyLearner(verbose = False, impact = 0.0005, commission=0)
    stratlearner3.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    strat_trades3 = stratlearner.testPolicy(symbol=symbol, sd=sd, ed=ed,sv=sv)
    strat_portval3 = ms.compute_portvals(symbol=symbol, orders_df=strat_trades3, start_val=sv, commission=0,impact=0.0005)

    # Normalize
    strat_portval = strat_portval / strat_portval[0]
    strat_portval2 = strat_portval2 / strat_portval2[0]
    strat_portval3 = strat_portval3 / strat_portval3[0]

    first_cr, first_sddr, first_adr = get_stats(strat_portval)
    second_cr, second_sddr, second_adr = get_stats(strat_portval2)
    third_cr, third_sddr, third_adr = get_stats(strat_portval3)


    # Referenced: https://stackoverflow.com/questions/36571560/directing-print-output-to-a-txt-file
    with open('images/ImpactAnalysisStats.txt', 'w') as file:
        print('.05 CR = ' + str(round(first_cr, 6)), file=file)
        print('.05 Stdev = ' + str(round(first_sddr, 6)), file=file)
        print('.05 Mean = ' + str(round(first_adr, 6)), file=file)
        print('.005 CR = ' + str(round(second_cr, 6)), file=file)
        print('.005 Stdev = ' + str(round(second_sddr, 6)), file=file)
        print('.005 Mean = ' + str(round(second_adr, 6)), file=file)
        print('.0005 CR = ' + str(round(third_cr, 6)), file=file)
        print('.0005 Stdev = ' + str(round(third_sddr, 6)), file=file)
        print('.0005 Mean = ' + str(round(third_adr, 6)), file=file)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strat_portval.index, strat_portval.values, color='green', label='.05 Impact')
    ax.plot(strat_portval2.index, strat_portval2.values, color='orange', label='.005 Impact')
    ax.plot(strat_portval3.index, strat_portval3.values, color='blue', label='.0005 Impact')
    plt.title('Strategy Learner Impact Analysis In-Sample (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Normalized)')
    plt.legend(loc='upper left')
    plt.grid(visible=True)
    plt.savefig('images/StrategyLearnerImpactAnalysis.png')
    plt.clf()