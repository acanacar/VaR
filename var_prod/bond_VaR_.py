from collections import namedtuple
import scipy.optimize as optimize
import pandas as pd
import numpy as np


def create_yield_return_series(mean=0, std=1):
    rate_days = [30, 90, 180, 270, 360, 720, 1080, 1440, 1800, 2160, 2520, 2880, 3240, 3600]
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
    df.interpolate(axis=1, inplace=True)
    return df


yield_returns = create_yield_return_series()

'''ONEMLI --- IMPORTANT'''


# 2 bononun portfoy halindeki riskini hesaplarken bonolarin ayni tarihnin riskinden ziyadae farkli tarihlerdeki riskleri portfoy yapilabilir.
# YANI, sen portfoyu olustururken 1 hissede bugunun riski ve digerinde or. 5 gun sonraki risk seklinde bir portfoy olusturabilirsin
# Bu da sana ayni gun icinde sona ererek neticeye erismis bir risk predictioni degil 5 gun sonra neticeye ulasan bir risk pozisyonu olusturur.
# Diyelim bir valigin riski 3 gunde az degisiyor.Digerinin 7 gunde. Bu sekilde bir portfoy yapilabilir.


class Bond(object):
    Coupon = namedtuple('Coupon', ['mtd', 'dtm', 'nominal', 'bond_code'])

    def __init__(self,
                 coupon_rate,
                 maturity,
                 face_value,
                 frequency,
                 settlement_date='2019-01-03'):
        self.bond_price = None
        self.coupons = None
        self.face_coupon = None
        self.remained_coupons = None
        self.bond_purchased_price = None
        self.yield_to_maturity = None
        self.bond_code = None
        self.portfolio_frame = None
        self.settlement_date = pd.to_datetime(settlement_date)
        self.current_date = pd.to_datetime('today').floor('D')
        self.current_date = self.settlement_date
        self.maturity = maturity
        self.coupon_rate = coupon_rate
        self.face_value = face_value
        self.frequency = frequency
        self.maturity_date = self.settlement_date + pd.DateOffset(days=self.maturity * 364)
        self.number_of_periods = self.maturity * self.frequency
        self.coupon_payment = self.coupon_rate * self.face_value / self.frequency
        self.create_bond_code()

    def create_bond_code(self):
        self.bond_code = 'TRB{}T{}{}'.format(
            self.maturity_date.strftime('%d%m%Y'),
            np.random.randint(100), np.random.randint(100))

    def create_coupons(self):
        ''' odenecek kuponlarin tarihleri ve odenme tarihlerine kalan gun sayilarini hesaplar '''
        mdts = pd.date_range(start=self.settlement_date,
                             end=self.maturity_date,
                             periods=self.number_of_periods + 1)
        dtms = list(map(lambda c_date: (c_date - self.current_date).days, mdts))
        self.coupons = [Bond.Coupon(mdt, dtm, self.coupon_payment, self.bond_code) for mdt, dtm in
                        zip(mdts, dtms) if
                        mdt > self.current_date]

    def create_face_coupon(self):
        dtm = (self.maturity_date - self.current_date).days
        self.face_coupon = Bond.Coupon(self.maturity_date, dtm, self.face_value, self.bond_code)

    def get_portfolio_frame(self):
        self.create_coupons()
        self.create_face_coupon()
        portfolio_frame = pd.DataFrame.from_records(self.coupons + [self.face_coupon],
                                                    columns=Bond.Coupon._fields)
        return portfolio_frame


    def calc_pv_of_face_value(self):
        """ maturity date'de odenecek face value nun present value sunu hesaplar """
        days_to_face_value = (self.maturity_date - self.current_date).days
        discount_rate = yield_returns.loc[self.current_date, days_to_face_value]
        return self.face_value / (1 + discount_rate)

    def check_remained_coupons(self):
        ''' vadesi gelmemis kuponlar belirlenir '''
        if self.current_date >= self.coupons[0][0]:
            self.remained_coupons = [coupon for coupon in self.coupons if coupon[0] > self.current_date]
        else:
            self.remained_coupons = self.coupons

    def price_to_bond(self):
        self.create_coupons()
        self.check_remained_coupons()
        price = 0
        for coupon in self.remained_coupons:
            discount_rate = yield_returns.loc[coupon[0], coupon[1]]
            price += self.coupon_payment / (1 + discount_rate)
        pv_face_value = self.calc_pv_of_face_value()
        price += pv_face_value
        self.bond_price = price

    def calc_ytm(self, guess=0.05):
        freq = self.frequency
        price = self.bond_purchased_price
        duration_terms = [(i + 1) for i in range(int(self.number_of_periods))]
        ytm_func = lambda y: \
            sum(
                [np.divide(self.coupon_payment, (1 + y / freq) ** t) for t in duration_terms]
            ) \
            + np.divide(self.face_value, (1 + y / freq) ** (freq * self.maturity)) \
            - price
        ytm = optimize.newton(ytm_func, guess)
        self.yield_to_maturity = ytm

# b1 = Bond(coupon_rate=.0575, maturity=1.5, face_value=100, frequency=2)
# b1.bond_purchased_price = 95.0428
# b1.price_to_bond()
# b1.portfolio_frame = b1.get_portfolio_frame()
