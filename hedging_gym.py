import gym
import numpy as np
from gym import error, spaces, utils

class HedgingEnv(gym.Env):
    def __init__(self, asset_price_model, option_price_model, max_steps=100,
                 option_contract_size=100, initial_holding_fraction=0.5,
                 trading_cost_parameter=0.01, trading_cost_gradient=0.01):

        self.asset_price_model  = asset_price_model
        self.option_price_model = option_price_model
        self.dt                 = option_price_model.dt
        self.T                  = option_price_model.T

        assert(asset_price_model.dt == option_price_model.dt, "dt mismatch")
        assert(self.dt * max_steps < self.T, "total steps of dt exceed T")

        # environment variables
        self.IHF = initial_holding_fraction
        self.OCS = option_contract_size
        self.TCP = trading_cost_parameter
        self.TCG = trading_cost_gradient

        # bookkeeping variables
        self.max_steps = max_steps
        self.num_steps = 0
        self.done      = False

        # state variables
        self.asset_price  = self.asset_price_model.get_asset_price()
        self.option_price = self.option_price_model.get_option_price(self.num_steps, self.asset_price)
        self.delta        = self.option_price_model.get_option_delta(self.num_steps, self.asset_price)
        self.h            = int(self.IHF * self.OCS)
        self.t            = self.T


    def _compute_pnl(self, delta_h, h, old_asset_price, new_asset_price,
                                       old_option_price, new_option_price):
        # pnl from imprefect hedging
        asset_pnl   = (-h/self.OCS) * (new_asset_price - old_asset_price)
        option_pnl  = new_option_price - old_option_price
        hedge_pnl   = asset_pnl + option_pnl

        # pnl from trading costs
        lots_traded = abs(delta_h)
        trading_pnl = -self.TCP * self.dt * (lots_traded + self.TCG * lots_traded**2)

        return hedge_pnl, trading_pnl

    def _state(self):
        return np.array([self.h, self.t, self.asset_price, self.option_price, self.delta], dtype=float)

    def step(self, new_h):
        assert(type(new_h) == int, "new_h must be an integer")

        # need these variables for reward computation
        old_h             = self.h
        old_asset_price   = self.asset_price
        old_option_price  = self.option_price

        # take a step
        self.asset_price_model.step()

        # bookkeeping variables
        self.num_steps += 1
        if self.num_steps == self.max_steps:
            self.done = True

        # state variables
        self.asset_price  = self.asset_price_model.get_asset_price()
        self.option_price = self.option_price_model.get_option_price(self.num_steps, self.asset_price)
        self.delta        = self.option_price_model.get_option_delta(self.num_steps, self.asset_price)
        self.h            = new_h
        self.t            = self.T - self.num_steps * self.dt

        pnls = self._compute_pnl(new_h - old_h, self.h, old_asset_price, self.asset_price,
                                                        old_option_price, self.option_price)
        state  = self._state()
        return state, pnls, self.done


    def reset(self):
        self.asset_price_model.reset()

        # bookkeeping variables
        self.num_steps = 0
        self.done      = False

        # state variables
        self.asset_price  = self.asset_price_model.get_asset_price()
        self.option_price = self.option_price_model.get_option_price(self.num_steps, self.asset_price)
        self.delta        = self.option_price_model.get_option_delta(self.num_steps, self.asset_price)
        self.h            = int(self.IHF * self.OCS)
        self.t            = self.T

        return self._state()

    def state_size(self):
        return len(self._state())
