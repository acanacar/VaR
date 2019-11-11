from constants import *
from scipy.stats import norm


def get_df(data, col):
    df = data.iloc[:, data.columns.get_level_values(0) == col]
    df.columns = df.columns.droplevel()

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any')

    returns = df.pct_change().iloc[1:]
    returns.sort_index()
    return df, returns


def show_plot(data, ticker):
    data[ticker].plot()
    plt.show()


def show_hist_plot(data, ticker):
    data[ticker].hist(bins=100)
    plt.show()


def get_portfolio(data,n=4):

    lis = np.random.rand(n)
    lis_sum = functools.reduce(lambda a, b: a + b, lis)
    weights = list(map(lambda y: y / lis_sum, lis))
    cols = random.sample(list(data.columns), k=n)

    return cols, weights


def VaR(Data, Returns, Method='Parametric Normal', Confidence_Interval=0.95, Period_Interval=None,
        EWMA_Discount_Factor=0.94, Series=False, removeNa=True):

    if Method == 'Historical_Simulation':

        if Series == False:
            Data = Returns[-Period_Interval:]
            Value_at_Risk = -np.percentile(Data, 1 - Confidence_Interval)
        if Series == True:
            series_name = '{}_{}_{}'.format(Method, Period_Interval, Confidence_Interval)
            Value_at_Risk = pd.Series(index=Returns.index, name=series_name)
            for i in range(0, len(Returns) - Period_Interval):
                if i == 0:
                    Data = Returns[-Period_Interval:]  # alttaki satirin nin daha sade matematiksel sekli
                    # Data = Returns[-Period_Interval+i:]
                else:
                    Data = Returns[-(Period_Interval + i):-i]
                Value_at_Risk[-i - 1] = -np.percentile(Data, 1 - Confidence_Interval)
    if Method == 'Parametric Normal':

        if Series == False:
            Data = Returns[-Period_Interval:]
            stdev = np.std(Data)
            Value_at_Risk = stdev * norm.ppf(Confidence_Interval)
        if Series == True:
            Value_at_Risk = pd.Series(index=Returns.index, name='ParVaR')
            for i in range(0, len(Returns) - Period_Interval):
                if i == 0:
                    Data = Returns[-(Period_Interval):]
                else:
                    Data = Returns[-(Period_Interval + i):-i]
                stdev = np.std(Data)
                Value_at_Risk[-i - 1] = stdev * norm.ppf(Confidence_Interval)
    if Method == 'Filtered Historical Simulation':

        # Defining exponentially smoothed weights components
        Degree_of_Freedom = np.empty([Period_Interval, ])
        Weights = np.empty([Period_Interval, ])
        Degree_of_Freedom[0] = 1.0
        Degree_of_Freedom[1] = EWMA_Discount_Factor
        Range = range(Period_Interval)
        for i in range(2, Period_Interval):
            Degree_of_Freedom[i] = Degree_of_Freedom[1] ** Range[i]
        for i in range(Period_Interval):
            Weights[i] = Degree_of_Freedom[i] / sum(Degree_of_Freedom)

        Value_at_Risk = pd.Series(index=Returns.index, name='FHSVaR')
        EWMAstdev = np.empty([len(Returns) - Period_Interval, ])
        stndrData = pd.Series(index=Returns.index)

        # For efficiency here we square returns first so the loop does not do it repeadetly
        sqrdReturns = Returns ** 2

        # Computations here happen in different times, because we first need all the EWMAstdev
        # First get the stdev according to the EWMA
        for i in range(0, len(Returns) - Period_Interval):
            if i == 0:
                sqrdData = sqrdReturns[-(Period_Interval):]
            else:
                sqrdData = sqrdReturns[-(Period_Interval + i):-i]

            EWMAstdev[-i - 1] = math.sqrt(sum(Weights * sqrdData))

        # Now get the Standardized data by dividing for the EWMAstdev.
        # Length is here -1 because we standardize by the EWMAstdev of the PREVIOUS period.
        # Hence also EWMAstdev is [-i-2] instead of [-i-1].
        for i in range(0, len(Returns) - Period_Interval - 1):
            stndrData[-i - 1] = Returns[-i - 1] / EWMAstdev[-i - 2]
        stndrData = stndrData[pd.notnull(stndrData)]
        # Finally get the percentile and unfilter back the data
        for i in range(0, len(stndrData) - Period_Interval):
            if i == 0:
                stndrData2 = stndrData[-(Period_Interval):]
            else:
                stndrData2 = stndrData[-(Period_Interval + i):-i]

            stndrData_pct = np.percentile(stndrData2, 1 - Confidence_Interval)
            # Unfilter back with the CURRENT stdev
            Value_at_Risk[-i - 1] = -(stndrData_pct * EWMAstdev[-i - 1])

        # For FHS the single take of VaR does not work because we need to standardize for the preceeding stdev
        # hence it is always necessary to calculate the whole series and take the last value
        if Series == True:
            Value_at_Risk = Value_at_Risk
        if Series == False:
            Value_at_Risk = Value_at_Risk[-1]
    return (Value_at_Risk)


