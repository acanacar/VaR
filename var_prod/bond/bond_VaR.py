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

