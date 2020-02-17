import pandas as pd

import numpy as np

parameters = {
    'coupon_payment': 30,
    'T': 50,
    'par_value': 1000,
    'yield_to_maturity': .03,
    'frequency': 1
}
bondPrice = 0
periods = parameters['T'] * parameters['frequency']
for i in range(1, parameters['T'] + 1):
    if i == parameters['T']:
        bondPrice += (parameters['coupon_payment'] + parameters['par_value']) / (
                1 + parameters['yield_to_maturity']) ** i
    else:
        bondPrice += parameters['coupon_payment'] / (1 + parameters['yield_to_maturity']) ** i

m = 1
t = 50
ytm = 0.03
fv = 1000
c = 0.03

bondPrice = ((fv * c / m * (1 - (1 + ytm / m) ** (-m * t))) / (ytm / m)) + fv * (1 + (ytm / m)) ** (-m * t)

print(bondPrice)

rate_days = [30, 90, 180, 270, 360, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240, 3600]


def create_yield_return_series(mean=0, std=1):
    l = []
    dates = pd.date_range(start='1/1/2016', end='1/01/2021')
    for day in range(rate_days[0], 3601):
        if day in rate_days:
            arr = np.random.normal(mean, std, size=len(dates))
            data = pd.Series(arr, index=dates, name=day)
        else:
            data = pd.Series(index=dates, name=day)
        l.append(data)

    df = pd.concat(l, axis=1)
    return df


d = create_yield_return_series()
d.interpolate(axis=1, inplace=True)


class Bond(object):
    def __init__(self,
                 coupon_rate,
                 maturity,
                 face_value,
                 yield_to_maturity,
                 frequency,
                 settlement_date='2019-01-03'):
        self.bond_price = None
        self.coupons = None
        self.remained_coupons = None
        self.settlement_date = pd.to_datetime(settlement_date)
        self.maturity = maturity
        self.coupon_rate = coupon_rate
        self.face_value = face_value
        self.yield_to_maturity = yield_to_maturity
        self.frequency = frequency
        self.maturity_date = self.settlement_date + pd.DateOffset(days=self.maturity * 364)
        self.current_date = pd.to_datetime('today').floor('D')
        self.number_of_periods = self.maturity * self.frequency
        self.coupon_payment = self.coupon_rate * self.face_value

    def calc_coupon_dates(self):
        coupon_dates = pd.date_range(start=self.settlement_date,
                                     end=self.maturity_date,
                                     periods=self.number_of_periods)
        coupon_remaining_days = list(map(lambda c_date: (c_date - self.current_date).days, coupon_dates))
        self.coupons = list(zip(coupon_remaining_days, coupon_dates))[1:]

    def calc_pv_of_face_value(self):
        face_value_remaining_days = (self.maturity_date - self.current_date).days
        discount_rate = d.loc[self.current_date[1], face_value_remaining_days]

        return self.face_value / (1 + discount_rate)

    def price_to_bond(self):
        if self.current_date >= self.coupons[0][0]:
            self.remained_coupons = [coupon for coupon in self.coupons if coupon[0] > self.current_date]
        else:
            self.remained_coupons = self.coupons
        price = 0
        for coupon in self.remained_coupons:
            discount_rate = d.loc[coupon[1], coupon[0]]
            price += self.coupon_payment / (1 + discount_rate)
        pv_face_value = self.calc_pv_of_face_value()
        price += pv_face_value
        self.bond_price = price
