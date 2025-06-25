import gymnasium as gym
import numpy as np
import pandas as pd

class StockPortfolioEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df_prices, df_sentiment, window_size=30, initial_capital=100000, transaction_cost_pct=0.001):
        super(StockPortfolioEnv, self).__init__()

        self.df_prices = df_prices
        self.df_sentiment = df_sentiment
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct

        self.tickers = df_prices.columns
        self.n_assets = len(self.tickers)
        
        self.df = self.df_prices.join(self.df_sentiment['sentiment']).ffill().dropna()
        self.start_tick = self.window_size
        self.end_tick = len(self.df) - 1

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.n_assets + self.n_assets + 1,),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_tick = self.start_tick
        self.portfolio_value = self.initial_capital
        self.portfolio_weights = np.zeros(self.n_assets)
        self.done = False
        
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        price_window = self.df.iloc[self.current_tick - self.window_size : self.current_tick][self.tickers]
        normalized_prices = price_window.values / price_window.iloc[0].values
        
        sentiment_score = self.df.iloc[self.current_tick - 1]['sentiment']
        
        obs = np.concatenate([
            normalized_prices.flatten(),
            self.portfolio_weights,
            np.array([sentiment_score])
        ])
        return obs.astype(np.float32)

    def _get_info(self):
        safe_tick = min(self.current_tick, self.end_tick)
        return {
            "portfolio_value": self.portfolio_value,
            "portfolio_weights": self.portfolio_weights,
            "timestamp": self.df.index[safe_tick]
        }

    def step(self, action):
        last_portfolio_value = self.portfolio_value
        target_weights = self._softmax(action)
        
        turnover = np.sum(np.abs(target_weights - self.portfolio_weights))
        transaction_costs = self.portfolio_value * turnover * self.transaction_cost_pct
        
        self.portfolio_weights = target_weights

        price_change_ratio = (self.df.iloc[self.current_tick][self.tickers].values / 
                              self.df.iloc[self.current_tick - 1][self.tickers].values)
        
        portfolio_return_ratio = np.dot(self.portfolio_weights, price_change_ratio)
        
        self.portfolio_value = self.portfolio_value * portfolio_return_ratio - transaction_costs
        
        reward = np.log(self.portfolio_value / last_portfolio_value)
        
        self.current_tick += 1
        
        if self.current_tick > self.end_tick:
            self.done = True
        
        if self.portfolio_value < self.initial_capital * 0.5:
            self.done = True

        return self._get_observation(), reward, self.done, False, self._get_info()
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def render(self, mode='human'):
        safe_tick = min(self.current_tick - 1, self.end_tick)
        date_str = self.df.index[safe_tick].strftime('%Y-%m-%d')
        print(f"Data: {date_str}")
        print(f"Valor do Portfólio: ${self.portfolio_value:,.2f}")
        print(f"Pesos: {self.portfolio_weights}")
        print("-" * 20)
