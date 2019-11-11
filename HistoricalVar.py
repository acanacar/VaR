from constants import *
from functions import *

# get all datas
store = pd.HDFStore(hist_store)

data = store['/all']
# swapping multiindex column levels
data.columns = data.columns.swaplevel(0, 1)

# Sadece Adj Close u iceren df ve adj close uzerinden daily return olarak hesaplanan returns olusturuldu
df, returns = get_df(data=data, col='Adj Close')

# randomly select securities ve weights
securities, weights = get_portfolio(data=df, n=4)

# adding return column whose securities and weights
returns['return'] = returns[securities].dot(weights)
returns['return'].name = 'return'
df_returns = returns['return'] * 100  # grafikteki daha guzel goruntu icin

period_intervals = [100, 252, 350, 500]
confidence_intervals = [.68, .95, .997, 1]
method = 'Historical_Simulation'

df_VaRs = []

df_VaRs.append(
    (df_returns, {'securities': securities,
                  'periodic_interval': 'return',
                  'confidence_interval': 'return',
                  'method': None})
)

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

for show_ci in confidence_intervals:
    All = pd.concat([df for df, title in df_VaRs if
                     title['confidence_interval'] == show_ci or title['confidence_interval'] == 'return'],
                    axis=1)
    plt.title(str(securities) + str(weights))
    All.plot(lw=1)
    png_name = 'Historical_{}'.format(show_ci)
    png_path = r'C:\Users\a.acar\PycharmProjects\VaR\outputs\{}.png'.format(png_name)
    plt.savefig(png_path)

'''
IID OLMALARINI GOZ ONUNDE BULUNDUR. BIR ONCEKI 500 GUN VE BI SONRAKI IID ISE,BIRINDEN DIGERINI YORUMLAYABILIRIZ.
BUYUK POPULASYONUN 2 ELEMANI GIBI.
PEKI 15 ER GUNLUK IID DAGILAN POPULASYONDAN N KERE 30 ELEMAN CEKEREK 500 ELEMANLIK BIR SAMPLE OLUSTURABILIR MIYIM YUKARIDAKI
MADDEYI SAGLAMASI ICIN

///////
BACKTESTTE HER HAKLI CIKTIGINI GUNU TOPLAYIP TOTAL BI FORWARD TEST HAVUZDA OLUSTURULABILIR
'''

'''Show VaR Graph for only one confidence interval
show_ci = .997
All = pd.concat([df for df, title in df_VaRs if
                 title['confidence_interval'] == show_ci or title['confidence_interval'] == 'return'],
                axis=1)

All.plot(lw=1)
plt.show()
'''

# show_plot(df, 'XU100.IS')
# show_hist_plot(df, 'XU100.IS')
