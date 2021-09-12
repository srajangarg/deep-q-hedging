import numpy as np
from abc import ABC, abstractmethod


class GenericAssetPriceModel(ABC):
    @abstractmethod
    def get_asset_price(self):
        pass

    @abstractmethod
    def step(self, *action):
        pass

    @abstractmethod
    def reset(self):
        pass


class GBM(GenericAssetPriceModel):
    def __init__(self, mu, vol, S_0, dt):
        self.mu  = mu
        self.dt  = dt
        self.S_0 = S_0
        self.vol = vol
        self.S_t = S_0

    def step(self):
        sample = np.random.normal(0, np.sqrt(self.dt))
        GBM_return = np.exp((self.mu - self.vol ** 2 / 2) * self.dt + self.vol * sample)
        self.S_t = self.S_t * GBM_return

    def reset(self):
        self.S_t = self.S_0

    def get_asset_price(self):
        return self.S_t