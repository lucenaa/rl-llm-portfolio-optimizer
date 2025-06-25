# Agente Híbrido (RL+LLM) para Otimização de Portfólio

Este repositório contém a implementação de um agente autônomo para otimização dinâmica de portfólios financeiros. O projeto utiliza uma abordagem híbrida, combinando Aprendizado por Reforço (RL) para a tomada de decisões estratégicas e um Modelo de Linguagem Grande (LLM) para extrair insights de notícias financeiras.

## 📜 Visão Geral

O objetivo deste projeto é desenvolver e avaliar um agente que aprende a gerenciar um portfólio de ativos (SPY, QQQ, GLD) de forma autônoma. Diferentemente de abordagens tradicionais, este agente enriquece sua percepção do mercado com uma análise de sentimento de notícias, realizada pelo Google Gemini, para tomar decisões mais informadas.

A estratégia aprendida pelo agente não foca em maximizar o lucro a qualquer custo, mas em otimizar o retorno ajustado pelo risco, buscando um crescimento de capital mais estável e com proteção contra grandes perdas.

## ✨ Principais Features

- **Agente Híbrido**: Sinergia entre o algoritmo PPO (Proximal Policy Optimization) do Aprendizado por Reforço e a capacidade de interpretação do Google Gemini.
- **Análise de Sentimento**: Extração de um score de sentimento de notícias financeiras para alimentar o estado do agente.
- **Ambiente de Backtesting Realista**: Simulação de mercado com custos de transação para uma avaliação mais precisa.
- **Análise de Performance**: Comparação de métricas financeiras chave (Sharpe Ratio, Max Drawdown, Retorno Total) contra um benchmark passivo (Buy & Hold).

## ⚙️ Arquitetura do Projeto

O fluxo de trabalho do projeto é dividido em cinco scripts sequenciais:

1. **`1_data_collection.py`**: Coleta de dados históricos de preços via yfinance e criação de um dataset simulado de notícias.
2. **`2_feature_extraction.py`**: Utilização da API do Google Gemini para processar as notícias e gerar um score de sentimento, salvando o resultado.
3. **`3_rl_environment.py`**: Definição do ambiente de simulação customizado (Gymnasium), onde o agente irá treinar e ser avaliado.
4. **`4_train_agent.py`**: Treinamento do agente PPO utilizando 80% dos dados históricos. O modelo treinado (o "cérebro" do agente) é salvo em um arquivo .zip.
5. **`5_evaluate_agent.py`**: Avaliação do agente treinado nos 20% de dados restantes (nunca vistos antes), gerando as métricas de desempenho e o gráfico final.

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Aprendizado por Reforço**: Stable Baselines3 (com PyTorch) & Gymnasium
- **Análise de Dados**: Pandas & NumPy
- **Coleta de Dados**: YFinance
- **Modelo de Linguagem**: Google Gemini API
- **Visualização**: Matplotlib

## 🚀 Instalação e Configuração

Siga os passos abaixo para configurar o ambiente e rodar o projeto localmente.

### 1. Clone o repositório:

```bash
git clone https://github.com/lucenaa/rl-llm-portfolio-optimizer
cd rl-llm-portfolio-optimizer
```

### 2. Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
# No Windows, use: venv\Scripts\activate
```

### 3. Instale as dependências:
(Certifique-se de que você criou o arquivo `requirements.txt`).

```bash
pip install -r requirements.txt
```

### 4. Configure sua chave de API:

- Crie um arquivo chamado `.env` na raiz do projeto.
- Adicione sua chave da API do Google Gemini a este arquivo, da seguinte forma:

```
GEMINI_API_KEY=SUA_CHAVE_API_AQUI
```

## ▶️ Como Executar

Execute os scripts na ordem correta. Cada script realiza uma etapa do processo.

### 1. Coletar dados de preços e notícias:

```bash
python3 1_data_collection.py
```

### 2. Extrair sentimento com o LLM (requer a chave de API):

```bash
python3 2_feature_extraction.py
```

### 3. Treinar o agente de RL (pode levar alguns minutos):

```bash
python3 4_train_agent.py
```

### 4. Avaliar o agente e gerar os resultados:

```bash
python3 5_evaluate_agent.py
```

## 📊 Resultados

O agente foi avaliado em um período de teste de ~8 meses (Mai/2023 a Jan/2024). Os resultados demonstram que, embora a estratégia passiva de Buy & Hold tenha gerado um retorno bruto maior durante este forte mercado de alta, o agente demonstrou uma eficiência de risco-retorno significativamente superior.

| Métrica | Agente RL+LLM | Benchmark (Buy & Hold SPY) |
|---------|---------------|---------------------------|
| **Retorno Total** | 17.86% | ~ 26% |
| **Maximum Drawdown** | -7.69% | ~ -8% |
| **Sharpe Ratio (Anual.)** | 2.33 | ~ 1.8 |

A principal conclusão é que o agente aprendeu com sucesso a priorizar a consistência e a gestão de risco, validando a abordagem híbrida proposta.

## 🔮 Trabalhos Futuros

- [ ] **Otimização de Hiperparâmetros**: Utilizar ferramentas como Optuna para encontrar a melhor combinação de parâmetros para o algoritmo PPO.
- [ ] **Features Avançadas de LLM**: Extrair informações mais complexas que apenas sentimento, como identificação de eventos ou tópicos macroeconômicos.
- [ ] **Avaliação em Diferentes Regimes**: Testar o desempenho do agente em dados de mercados de baixa (bear market) ou laterais.
- [ ] **Modelagem de Risco Alternativa**: Explorar funções de recompensa baseadas em outras métricas, como o Sortino Ratio.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.
