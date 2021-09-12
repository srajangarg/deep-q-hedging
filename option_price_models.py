import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
import warnings


class GenericOptionPriceModel(ABC):
    """
    Generic option price class. The use this with the gym-hedging environment, the option class needs to have
    a function called 'compute_option_price' that computes the option price for a given the asset price and

    """
    @abstractmethod
    def get_option_price(self, *inputs):
        pass


class BSM(GenericOptionPriceModel):
    def __init__(self, strike_price, rf_interest_rate, volatility, T, dt):
        self.T         = T
        self.K         = strike_price
        self.r         = rf_interest_rate
        self.vol       = volatility
        self.dt        = dt

    def get_time_to_maturity(self, step_num):
        return self.T - step_num * self.dt

    def get_option_price(self, step_num, asset_price):
        self.t = self.get_time_to_maturity(step_num)
        self.S = asset_price
        assert(self.t > 1e-6, "time too close to maturirty")

        d_1 = self._d1()
        d_2 = d_1 - self.vol * np.sqrt(self.t)
        return norm.cdf(d_1) * self.S - norm.cdf(d_2) * self.K * np.exp(-self.r * self.t)

    def get_option_delta(self, step_num, asset_price):
        self.t = self.get_time_to_maturity(step_num)
        self.S = asset_price
        return norm.cdf(self._d1())


    def _d1(self):
        return (np.log(self.S / self.K) + (self.r + self.vol**2 / 2) * self.t) / (self.vol * np.sqrt(self.t))



