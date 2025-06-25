import pandas as pd
import yfinance as yf
import os

TICKERS = ["SPY", "GLD", "QQQ"]
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
PRICE_DATA_FILE = "stock_prices.csv"
NEWS_DATA_FILE = "simulated_news.csv"

def download_price_data():
    if os.path.exists(PRICE_DATA_FILE):
        print(f"Arquivo de preços '{PRICE_DATA_FILE}' já existe. Pulando o download.")
        return
    
    print(f"Baixando dados de preços para {TICKERS}...")
    
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)
    
    adj_close = data['Adj Close']
    adj_close.to_csv(PRICE_DATA_FILE)
    print(f"Dados de preços salvos em '{PRICE_DATA_FILE}'.")

def create_simulated_news():
    if os.path.exists(NEWS_DATA_FILE):
        print(f"Arquivo de notícias '{NEWS_DATA_FILE}' já existe. Pulando a criação.")
        return

    print("Criando arquivo de notícias simuladas...")
    news_data = {
        'date': pd.to_datetime([
            "2019-05-05", "2020-03-12", "2020-08-27", "2021-02-01",
            "2021-11-08", "2022-05-04", "2022-09-13", "2023-03-10",
            "2023-07-26"
        ]),
        'headline': [
            "US-China trade tensions escalate, markets tumble.",
            "WHO declares COVID-19 a pandemic, causing global market panic.",
            "Federal Reserve announces new inflation policy, signaling tolerance for higher rates.",
            "Retail trading frenzy drives massive volatility in certain stocks.",
            "Infrastructure bill passes in the US, boosting industrial and material sectors.",
            "Fed raises interest rates by 50 basis points to combat inflation.",
            "Higher-than-expected inflation report leads to worst market drop since 2020.",
            "Silicon Valley Bank collapses, sparking fears of a new banking crisis.",
            "Federal Reserve hikes interest rates again but hints at a potential pause."
        ]
    }
    df_news = pd.DataFrame(news_data)
    df_news.set_index('date', inplace=True)
    df_news.to_csv(NEWS_DATA_FILE)
    print(f"Notícias simuladas salvas em '{NEWS_DATA_FILE}'.")

if __name__ == "__main__":
    download_price_data()
    create_simulated_news()
