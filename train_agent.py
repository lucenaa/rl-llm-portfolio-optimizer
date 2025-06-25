import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import StockPortfolioEnv
import os

PRICE_DATA_FILE = "stock_prices.csv"
SENTIMENT_DATA_FILE = "news_with_sentiment.csv"
TRAINED_MODEL_NAME = "ppo_portfolio_manager"
TOTAL_TIMESTEPS = 50000

def train_agent():
    if not all(os.path.exists(f) for f in [PRICE_DATA_FILE, SENTIMENT_DATA_FILE]):
        print("Arquivos de dados não encontrados. Execute os scripts 1 e 2 primeiro.")
        return

    df_prices = pd.read_csv(PRICE_DATA_FILE, index_col='Date', parse_dates=True)
    df_sentiment = pd.read_csv(SENTIMENT_DATA_FILE, index_col='date', parse_dates=True)

    train_size = int(len(df_prices) * 0.8)
    df_prices_train = df_prices.iloc[:train_size]
    
    env = DummyVecEnv([lambda: StockPortfolioEnv(df_prices_train, df_sentiment)])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_portfolio_tensorboard/")

    print("Iniciando o treinamento do agente PPO...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    print("Treinamento concluído.")

    model.save(TRAINED_MODEL_NAME)
    print(f"Modelo salvo como '{TRAINED_MODEL_NAME}.zip'")

if __name__ == "__main__":
    train_agent()
