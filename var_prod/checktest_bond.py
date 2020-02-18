import pandas as pd
from constants import *
from functions import *
from var_prod.var_class import *
from var_prod.bond_VaR_ import *


def get_yield_returns(data, current_date):
    '''bononun sahip oldugu coupon ve face_coupon getirilerine ait vadelere denk gelen historical yield returnler olusturuluyor'''
    maturities = data['dtm'].unique()
    return yield_returns.loc[yield_returns.index < current_date, maturities].copy()


b1 = Bond(coupon_rate=.0575, maturity=1.5, face_value=100, frequency=2)
b1.bond_purchased_price = 95.0428
b1.price_to_bond()
b1.portfolio_frame = b1.get_portfolio_frame()
yield_df = get_yield_returns(data=b1.portfolio_frame, current_date=b1.current_date)
weights = (b1.portfolio_frame['nominal']/b1.portfolio_frame['nominal'].sum()).values

num_simulations = 10000
time_scaler = 1
calc_type = 'log'
period_interval = 252
confidence_interval = .95
price_col = 'Adj Close'
lambda_decay = .98


d = ParametricVaR(interval=confidence_interval,
                  weights=weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval,
                  timeScaler=time_scaler,
                  matrix=None,
                  returnMatrix=yield_df.values)
'''
d = HistoricalVaR(interval=confidence_interval,
                  matrix=input_df,
                  weights=portfolio_securities_weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval
                  )
d = HistoricalVaR(interval=confidence_interval,
                  matrix=input_df,
                  weights=portfolio_securities_weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval,
                  hybrid=True,
                  lambda_decay_hist=lambda_decay
                  )
d = ValueAtRisk(interval=confidence_interval,
                matrix=input_df,
                weights=portfolio_securities_weights,
                return_method=calc_type,
                lookbackWindow=period_interval,
                timeScaler=time_scaler)
'''
d = ParametricVaREwma(interval=confidence_interval,
                      matrix=input_df,
                      weights=portfolio_securities_weights,
                      return_method=calc_type,
                      lookbackWindow=period_interval,
                      timeScaler=time_scaler,
                      lambdaDecay=lambda_decay)
'''

d = MonteCarloVaR(interval=confidence_interval,
                  matrix=input_df,
                  weights=portfolio_securities_weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval,
                  numSimulations=num_simulations)
'''
d.vaR(series=True)
