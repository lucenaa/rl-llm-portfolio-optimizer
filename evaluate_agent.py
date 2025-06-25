import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rl_environment import StockPortfolioEnv
import os

PRICE_DATA_FILE = "stock_prices.csv"
SENTIMENT_DATA_FILE = "news_with_sentiment.csv"
TRAINED_MODEL_NAME = "ppo_portfolio_manager"

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(252) # Anualizado

def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values.cummax()
    drawdown = (portfolio_values - peak) / peak
    return drawdown.min()

def evaluate_agent():
    if not os.path.exists(f"{TRAINED_MODEL_NAME}.zip"):
        print(f"Modelo '{TRAINED_MODEL_NAME}.zip' não encontrado. Treine o agente primeiro.")
        return

    df_prices = pd.read_csv(PRICE_DATA_FILE, index_col='Date', parse_dates=True)
    df_sentiment = pd.read_csv(SENTIMENT_DATA_FILE, index_col='date', parse_dates=True)

    train_size = int(len(df_prices) * 0.8)
    df_prices_test = df_prices.iloc[train_size:]

    env_test = StockPortfolioEnv(df_prices_test, df_sentiment)
    
    model = PPO.load(TRAINED_MODEL_NAME, env=env_test)

    print("Iniciando avaliação no conjunto de teste...")
    obs, _ = env_test.reset()
    done = False
    
    portfolio_values = [env_test.initial_capital]
    timestamps = [env_test.df.index[env_test.start_tick]]
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_test.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        timestamps.append(info['timestamp'])

    print("Avaliação concluída.")

    df_results = pd.DataFrame({'timestamp': timestamps, 'portfolio_value': portfolio_values})
    df_results = df_results.set_index('timestamp')
    
    df_results['daily_return'] = df_results['portfolio_value'].pct_change()

    total_return = (df_results['portfolio_value'].iloc[-1] / df_results['portfolio_value'].iloc[0]) - 1
    sharpe_ratio = calculate_sharpe_ratio(df_results['daily_return'].dropna())
    max_drawdown = calculate_max_drawdown(df_results['portfolio_value'])

    print("\n--- MÉTRICAS DE DESEMPENHO (Conjunto de Teste) ---")
    print(f"Retorno Total: {total_return:.2%}")
    print(f"Sharpe Ratio (Anualizado): {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print("---------------------------------------------------\n")

    buy_and_hold_spy = df_prices_test['SPY'].copy()
    buy_and_hold_spy_values = (buy_and_hold_spy / buy_and_hold_spy.iloc[0]) * env_test.initial_capital
    df_results['buy_hold_spy'] = buy_and_hold_spy_values

    plt.figure(figsize=(14, 7))
    plt.plot(df_results.index, df_results['portfolio_value'], label='Agente RL+LLM')
    plt.plot(df_results.index, df_results['buy_hold_spy'], label='Buy & Hold (SPY)')
    plt.title('Desempenho do Agente vs. Buy & Hold')
    plt.xlabel('Data')
    plt.ylabel('Valor do Portfólio ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_chart.png")
    print("Gráfico de desempenho salvo como 'performance_chart.png'")
    plt.show()

if __name__ == "__main__":
    evaluate_agent()
