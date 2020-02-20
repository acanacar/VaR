from var_prod.var_class import *
from var_prod.bond_VaR_ import *

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.max_colwidth = 15


def get_yield_returns(data, current_date):
    '''bononun sahip oldugu coupon ve face_coupon getirilerine ait vadelere denk gelen historical yield returnler olusturuluyor'''
    maturities = data['dtm'].unique()
    return yield_returns.loc[yield_returns.index < current_date, maturities].copy()


b1 = Bond(coupon_rate=.0575, maturity=1.5, face_value=100, frequency=2)
b1.bond_purchased_price = 95.0428
b1.price_to_bond()
b1.portfolio_frame = b1.get_portfolio_frame()
yield_returns_df = get_yield_returns(data=b1.portfolio_frame, current_date=b1.current_date)
weights = b1.portfolio_frame.groupby('dtm').agg({'nominal': 'sum'}).apply(lambda x: x / float(x.sum())).values
weights = weights.flatten()

num_simulations = 10000
time_scaler = 1
calc_type = 'log'
period_interval = 252
confidence_interval = .95
price_col = 'Adj Close'
lambda_decay = .98

# d = ParametricVaR(interval=confidence_interval,
#                   weights=weights,
#                   return_method=calc_type,
#                   lookbackWindow=period_interval,
#                   timeScaler=time_scaler,
#                   matrix=yield_returns_df)
d = ParametricVaREwma(interval=confidence_interval,
                      weights=weights,
                      return_method=calc_type,
                      lookbackWindow=period_interval,
                      timeScaler=time_scaler,
                      lambdaDecay=lambda_decay,
                      matrix=yield_returns_df)
d.vaR()

# calculate bono VaR
rate_change_expectation = d.ValueAtRisk

b1.portfolio_frame['tomorrow_rates'] = (b1.portfolio_frame['current_rates'] * (1 + rate_change_expectation))
b1.portfolio_frame['new_dr'] = 1 / (1 + b1.portfolio_frame['tomorrow_rates'])
b1.portfolio_frame['new_pv'] = b1.portfolio_frame['nominal'] * b1.portfolio_frame['new_dr']
total_start_val = sum(b1.portfolio_frame['nominal'])

VaR_BOND = sum(b1.portfolio_frame['new_pv'] - b1.portfolio_frame['pv'])
New_val_bond = sum(b1.portfolio_frame['nominal'] / (1 + d.ValueAtRisk))
print('x')
