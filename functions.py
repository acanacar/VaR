from constants import *


def VaR_Compare(Returns, Confidence_Interval=0.95, Period_Interval=100, EWMA_Discount_Factor=0.94):
    'This function calculates different VaR series and plots it in the same graph'

    # Use for each VaR call the same values, here they are set
    Ret = Returns
    CI = Confidence_Interval
    PI = Period_Interval
    EWMAdf = EWMA_Discount_Factor

    # Call the single VaR series
    VaRPN = VaR(Ret, Formula='Parametric Normal', Confidence_Interval=CI, Period_Interval=PI,
                EWMA_Discount_Factor=EWMAdf, Series=True, removeNa=True)
    VaREWMA = VaR(Ret, Formula='Parametric EWMA', Confidence_Interval=CI, Period_Interval=PI,
                  EWMA_Discount_Factor=EWMAdf, Series=True, removeNa=True)
    VaRHS = VaR(Ret, Formula='Historical Simulation', Confidence_Interval=CI, Period_Interval=PI,
                EWMA_Discount_Factor=EWMAdf, Series=True, removeNa=True)
    VaRFHS = VaR(Ret, Formula='Filtered Historical Simulation', Confidence_Interval=CI, Period_Interval=PI,
                 EWMA_Discount_Factor=EWMAdf, Series=True, removeNa=True)

    # Concat the different VaR series in the same dataframe and plot it
    AllVaR = pd.concat([VaRPN, VaREWMA, VaRHS, VaRFHS], axis=1)
    AllVaR.plot(lw=1)

    return (AllVaR)


def get_df(data, col):
    df = data.iloc[:, data.columns.get_level_values(0) == col]
    df.columns = df.columns.droplevel()

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any')

    df = df.pct_change().iloc[1:]
    df.sort_index()
    return df


def show_plot(data, ticker):
    data[ticker].plot()
    plt.show()


def show_hist_plot(data, ticker):
    data[ticker].hist(bins=100)
    plt.show()


def get_portfolio(data):
    n = 4

    lis = np.random.rand(n)
    lis_sum = functools.reduce(lambda a, b: a + b, lis)
    weights = list(map(lambda y: y / lis_sum, lis))
    cols = random.sample(list(data.columns), k=n)

    return cols, weights


def VaR(Data,Returns, Method='Parametric Normal', Confidence_Interval=0.95, Period_Interval=None,
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
    return (Value_at_Risk)
