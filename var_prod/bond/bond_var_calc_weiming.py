# Calculating the yield to maturity¶

import scipy.optimize as optimize


def bond_ytm(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T * 2
    coupon = coup / 100. * par
    dt = [(i + 1) / freq for i in range(int(periods))]
    ytm_func = lambda y: \
        sum([coupon / freq / (1 + y / freq) ** (freq * t) for t in dt]) + \
        par / (1 + y / freq) ** (freq * T) - price

    return optimize.newton(ytm_func, guess)


ytm = bond_ytm(price=95.0428, par=100, T=1.5, coup=5.75, freq=2)


# Calculating the price of a bond¶

def bond_price(par, T, ytm, coup, freq=2):
    freq = float(freq)
    periods = T * 2
    coupon = coup / 100. * par
    dt = [(i + 1) / freq for i in range(int(periods))]
    price = sum([coupon / freq / (1 + ytm / freq) ** (freq * t) for t in dt]) + \
            par / (1 + ytm / freq) ** (freq * T)
    return price


bond_price(par=100,
           T=1.5,
           ytm=ytm,
           coup=5.75,
           freq=2)


# Bond duration¶
def bond_mod_duration(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy
    price_minus = bond_price(par, T, ytm_minus, coup, freq)

    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)

    mduration = (price_minus - price_plus) / (2 * price * dy)
    return mduration


mod_duration = bond_mod_duration(95.0428, 100, 1.5, 5.75, 2)
print(mod_duration)


# Bond convexity
def bond_convexity(price, par, T, coup, freq, dy=0.01):
    ytm = bond_ytm(price, par, T, coup, freq)

    ytm_minus = ytm - dy
    price_minus = bond_price(par, T, ytm_minus, coup, freq)

    ytm_plus = ytm + dy
    price_plus = bond_price(par, T, ytm_plus, coup, freq)

    convexity = (price_minus + price_plus - 2 * price) / (price * dy ** 2)
    return convexity
