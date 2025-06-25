import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Chave de API do Gemini não encontrada. Verifique seu arquivo .env")
genai.configure(api_key=API_KEY)

NEWS_FILE = "simulated_news.csv"
OUTPUT_FILE = "news_with_sentiment.csv"

def get_sentiment_from_gemini(text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Analyze the sentiment of the following financial news headline.
    Return only a single floating-point number from -1.0 (extremely negative) to 1.0 (extremely positive).
    Do not include any other text or explanation.

    Headline: "{text}"
    Sentiment Score:
    """
    
    try:
        response = model.generate_content(prompt)
        sentiment_str = ''.join(c for c in response.text if c in '-.0123456789')
        return float(sentiment_str.strip())
    except Exception as e:
        print(f"Erro ao chamar a API do Gemini: {e}")
        return 0.0

def process_news_file():
    if not os.path.exists(NEWS_FILE):
        print(f"Arquivo '{NEWS_FILE}' não encontrado. Execute '1_data_collection.py' primeiro.")
        return

    if os.path.exists(OUTPUT_FILE):
        print(f"Arquivo de sentimento '{OUTPUT_FILE}' já existe. Pulando processamento.")
        return

    df = pd.read_csv(NEWS_FILE, index_col='date', parse_dates=True)
    sentiments = []

    print("Processando notícias com o Gemini para extrair sentimento...")
    for index, row in df.iterrows():
        headline = row['headline']
        print(f"Analisando: '{headline}'")
        sentiment = get_sentiment_from_gemini(headline)
        sentiments.append(sentiment)
        time.sleep(1)

    df['sentiment'] = sentiments
    df.to_csv(OUTPUT_FILE)
    print(f"Sentimentos salvos em '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    process_news_file()
