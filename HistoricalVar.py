from constants import *
from functions import *

store = pd.HDFStore(hist_store)

data = store['/all']
data.columns = data.columns.swaplevel(0, 1)

df = get_df(data=data, col='Adj Close')

securities, weights = get_portfolio(data=df)




df['return'] = df[securities].dot(weights)
df['return'].name = 'return'
df_returns = df['return'] * 100

period_intervals = [100, 252, 350, 500]
confidence_intervals = [.68, .95, .997, 1]
method = 'Historical_Simulation'


#mean-variance
mean_var_period=252
mean_var_risk = pd.Series(index=df.index)
for i in range(0,len(df)-mean_var_period):
    if i==0:
        Data = df[securities][-mean_var_period:]
mean_var_returns = (pow(df[securities].iloc[-1] / df[securities].iloc[0], 1 / len(df)) - 1) * mean_var_period


df_VaRs = [
    (df_returns, {'securities': securities,
                  'periodic_interval': 'return',
                  'confidence_interval': 'return',
                  'method': None})
]

for p_i_ in period_intervals:
    for c_i_ in confidence_intervals:
        df_VaR = VaR(Data=df,
                     Returns=df_returns,
                     Method=method,
                     Confidence_Interval=c_i_,
                     Period_Interval=p_i_,
                     Series=True)
        title = {'securities': securities,
                 'periodic_interval': p_i_,
                 'confidence_interval': c_i_,
                 'method': method}
        df_VaRs.append(
            (df_VaR, title)
        )

show_ci = .997
All = pd.concat([df for df, title in df_VaRs if
                 title['confidence_interval'] == show_ci or title['confidence_interval'] == 'return'],
                axis=1)

All.plot(lw=1)
plt.show()

# show_plot(df, 'XU100.IS')
# show_hist_plot(df, 'XU100.IS')
'''
IID OLMALARINI GOZ ONUNDE BULUNDUR. BIR ONCEKI 500 GUN VE BI SONRAKI IID ISE,BIRINDEN DIGERINI YORUMLAYABILIRIZ.
BUYUK POPULASYONUN 2 ELEMANI GIBI.
PEKI 15 ER GUNLUK IID DAGILAN POPULASYONDAN N KERE 30 ELEMAN CEKEREK 500 ELEMANLIK BIR SAMPLE OLUSTURABILIR MIYIM YUKARIDAKI
MADDEYI SAGLAMASI ICIN

///////
BACKTESTTE HER HAKLI CIKTIGINI GUNU TOPLAYIP TOTAL BI FORWARD TEST HAVUZDA OLUSTURULABILIR
'''
