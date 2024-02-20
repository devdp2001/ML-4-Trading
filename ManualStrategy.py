import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data
import indicators as ind
import marketsimcode as ms

class ManualStrategy(object):

    def __init__(self):
        self.short_days = []
        self.long_days = []

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "dpatel426"  # Change this to your user ID

    def testPolicy(self, symbol= 'JPM', sd= dt.datetime(2008, 1, 1), ed = dt.datetime(2009,12,31), sv = 100000):
        prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')

        sma = ind.Simple_moving_Average(prices, 10)
        momentum = ind.Momentum(prices, 10)
        bbp = ind.Bollinger_Bands_Percentage(prices, 10)

        df_trades = prices.copy()
        df_trades[:] = 0
        holdings = 0

        for date, _ in prices.iterrows():
            buy_condition = (sma.loc[date].values[0] > prices.loc[date].values[0]).any() and (momentum.loc[date].values[0] < -.15).any() and (bbp.loc[date].values[0] < .2).any()
            sell_condition = (sma.loc[date].values[0] < prices.loc[date].values[0]).any() and (momentum.loc[date].values[0] > -.15).any() and (bbp.loc[date].values[0] > .8).any()
            if(buy_condition):
                if (holdings == 0):
                    df_trades.loc[date] = 1000
                    holdings += 1000
                    self.long_days.append(date)
                elif (holdings == -1000):
                    df_trades.loc[date] = 2000
                    holdings += 2000
                    self.long_days.append(date)
                else: df_trades.loc[date] = 0
            elif (sell_condition):
                if (holdings == 0):
                    df_trades.loc[date] = -1000
                    holdings -= 1000
                    self.short_days.append(date)
                elif (holdings == 1000):
                    df_trades.loc[date] = -2000
                    holdings -= 2000
                    self.short_days.append(date)
                else: df_trades.loc[date] = 0
        return df_trades

    def getshortlongdays(self):
        return self.short_days, self.long_days

    def get_stats(self, port_val):
        daily_rets = (port_val / port_val.shift(1) - 1)
        cr = (port_val[-1] / port_val[0]) - 1
        sddr = daily_rets.std()
        adr = daily_rets.mean()
        return cr, sddr, adr

    def runman(self):
        learner = ManualStrategy()

        #In-Sample
        prices = get_data(['JPM'], pd.date_range((dt.datetime(2008, 1, 1)), (dt.datetime(2009, 12, 31))),
                          addSPY=False).dropna(how='all')
        #Benchmark portval
        symbol = 'JPM'
        sv = 100000
        sd = dt.datetime(2008, 1, 1)
        ed = dt.datetime(2009, 12, 31)
        benchmarkJPM = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')
        benchmarkJPM[:] = 0
        benchmarkJPM.iloc[0, 0] = 1000
        benchmark_portval = ms.compute_portvals(symbol='JPM', orders_df=benchmarkJPM, start_val=sv, commission=9.95, impact=0.005)

        #Manual Strategy Portval
        manual_trades = learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
        manual_portval = ms.compute_portvals(symbol='JPM', orders_df=manual_trades, start_val=100000, commission=9.95, impact=0.005)


        # Normalize
        benchmark_portval = benchmark_portval / benchmark_portval[0]
        manual_portval = manual_portval / manual_portval[0]

        # Stats
        benchmark_cr, benchmark_sddr, benchmark_adr = learner.get_stats(benchmark_portval)
        manual_cr, manual_sddr, manual_adr = learner.get_stats(manual_portval)

        # Referenced: https://stackoverflow.com/questions/36571560/directing-print-output-to-a-txt-file
        with open('images/Benchmark_Vs_Manual_InSample.txt', 'w') as file:
            print('Benchmark CR = ' + str(round(benchmark_cr, 6)), file=file)
            print('Benchmark Stdev = ' + str(round(benchmark_sddr, 6)), file=file)
            print('Benchmark Mean = ' + str(round(benchmark_adr, 6)), file=file)
            print('Manual CR = ' + str(round(manual_cr, 6)), file=file)
            print('Manual Stdev = ' + str(round(manual_sddr, 6)), file=file)
            print('Manual Mean = ' + str(round(manual_adr, 6)), file=file)

        short_days, long_days = learner.getshortlongdays()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(benchmark_portval.index, benchmark_portval.values, color='purple', label='Benchmark')
        ax.plot(manual_portval.index, manual_portval.values, color='red', label='Manual')
        for date in short_days:
            ax.axvline(date, color='blue', linestyle='--', alpha=0.5)
        for date in long_days:
            ax.axvline(date, color='black', linestyle='--', alpha=0.5)
        ax.plot([], [], color='blue', linestyle='--', label='Long')
        ax.plot([], [], color='black', linestyle='--', label='Short')
        plt.title('Benchmark vs. Manual Portfolio Value for JPM In-Sample (Normalized)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (Normalized)')
        plt.legend(loc='upper left')
        plt.grid(visible=True)
        plt.savefig('images/ManualVsBenchInSample.png')
        plt.clf()

        #Out-Sample
        learner2 = ManualStrategy()
        prices = get_data(['JPM'], pd.date_range((dt.datetime(2010, 1, 1)), (dt.datetime(2011, 12, 31))),
                          addSPY=False).dropna(how='all')
        #Benchmark portval
        symbol = 'JPM'
        sv = 100000
        sd = dt.datetime(2010, 1, 1)
        ed = dt.datetime(2011, 12, 31)
        benchmarkJPM = get_data([symbol], pd.date_range(sd, ed), addSPY=False).dropna(how='all')
        benchmarkJPM[:] = 0
        benchmarkJPM.iloc[0, 0] = 1000
        benchmark_portval = ms.compute_portvals(symbol='JPM', orders_df=benchmarkJPM, start_val=sv, commission=9.95, impact=0.005)

        #Manual Strategy Portval
        manual_trades = learner2.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
        manual_portval = ms.compute_portvals(symbol='JPM', orders_df=manual_trades, start_val=100000, commission=9.95, impact=0.005)


        # Normalize
        benchmark_portval = benchmark_portval / benchmark_portval[0]
        manual_portval = manual_portval / manual_portval[0]

        # Stats
        benchmark_cr, benchmark_sddr, benchmark_adr = learner.get_stats(benchmark_portval)
        manual_cr, manual_sddr, manual_adr = learner.get_stats(manual_portval)

        # Referenced: https://stackoverflow.com/questions/36571560/directing-print-output-to-a-txt-file
        with open('images/Benchmark_Vs_Manual_OutSample.txt', 'w') as file:
            print('Benchmark CR = ' + str(round(benchmark_cr, 6)), file=file)
            print('Benchmark Stdev = ' + str(round(benchmark_sddr, 6)), file=file)
            print('Benchmark Mean = ' + str(round(benchmark_adr, 6)), file=file)
            print('Manual CR = ' + str(round(manual_cr, 6)), file=file)
            print('Manual Stdev = ' + str(round(manual_sddr, 6)), file=file)
            print('Manual Mean = ' + str(round(manual_adr, 6)), file=file)

        short_days, long_days = learner2.getshortlongdays()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(benchmark_portval.index, benchmark_portval.values, color='purple', label='Benchmark')
        ax.plot(manual_portval.index, manual_portval.values, color='red', label='Manual')
        for date in short_days:
            ax.axvline(date, color='blue', linestyle='--', alpha=0.5)
        for date in long_days:
            ax.axvline(date, color='black', linestyle='--', alpha=0.5)
        ax.plot([], [], color='blue', linestyle='--', label='Long')
        ax.plot([], [], color='black', linestyle='--', label='Short')
        plt.title('Benchmark vs. Manual Portfolio Value for JPM Out-Sample (Normalized)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (Normalized)')
        plt.legend(loc='upper left')
        plt.grid(visible=True)
        plt.savefig('images/ManualVsBenchOutSample.png')
        plt.clf()



if __name__ == "__main__":
    #testPolicy(symbol= 'JPM', sd= dt.datetime(2008, 1, 1), ed = dt.datetime(2009,12,31), sv = 100000)
    learner = ManualStrategy()
    #learner.run()
    #learner.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

