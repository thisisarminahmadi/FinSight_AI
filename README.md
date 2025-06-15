# FinSight AI: ETF Portfolio Recommendation Prototype

**Not Production Ready**

This repository contains a self-initiated prototype developed by Armin Ahmadi during the Charles Schwab interview process. It is intended for demonstration and evaluation purposes only, and is not production ready.

## Overview

FinSight AI is an AI-powered ETF portfolio recommendation system. It leverages natural language processing and robust financial data analysis to suggest ETF portfolios tailored to user preferences, including investment themes, budget, and risk tolerance.

## Two Modes: Offline and Live Data

This project provides two main scripts for ETF recommendations:

- **Offline Mode (`schwab_langchain_chatbot.py`)**: Uses a static ETF dataset (`etf_data.csv`). This CSV was generated using Gemini AI, with few-shot prompting and internet access to gather and structure ETF data. This mode is suitable for demonstration without relying on live data sources.

- **Live Data Mode (`schwab_langchain_chatbot_LiveYahoo.py`)**: Fetches real-time ETF data using the Yahoo Finance API. This mode demonstrates integration with live financial data, but is still a prototype and not intended for production use.

Both scripts provide a Gradio-based user interface for natural language queries and portfolio recommendations.

### Key Features
- **Natural Language Understanding:** Extracts investment themes, budget, and risk tolerance from user queries.
- **Portfolio-Level Theme Coverage:** Ensures the recommended portfolio covers all requested themes.
- **Smart ETF Selection:** Selects top ETFs for each theme, then fills remaining slots with best-scoring options, avoiding duplicates.
- **Markdown Table Output:** Presents recommendations in a clean, readable Markdown table.
- **Portfolio Metrics:** Calculates and displays average return, expense ratio, volatility, and sector concentration.
- **LLM Integration:** Uses OpenAI's GPT models for intent extraction and concise, formatted portfolio explanations.
- **User Experience:** Provides warnings for negative performance, clear error messages, and actionable suggestions.
- **Live Data (Optional):** The `schwab_langchain_chatbot_LiveYahoo.py` script can fetch real-time ETF data from Yahoo Finance.

## Usage

1. **Set up environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `OPENAI_MODEL`: GPT model to use (default: gpt-4.1)
   - `MAX_ETFS`: Maximum ETFs per portfolio (default: 4)
   - `MIN_ALLOC`: Minimum allocation per ETF in USD (default: 1000)
   - (Optional) `ETF_CSV`: Path to your ETF data CSV file (for the offline script)

2. **Run the main script:**
   - For live Yahoo Finance data: `python schwab_langchain_chatbot_LiveYahoo.py`
   - For offline data: `python schwab_langchain_chatbot.py`

3. **Interact via Gradio UI:**
   - Enter queries like:
     - "I want to invest $15,000 in clean energy and AI, medium risk"
     - "Can you suggest a low-risk portfolio for $10,000 in healthcare?"

## Data Files
- `ETF_Data_Gemeni2.5.csv`: Example ETF dataset used for static recommendations.
- `data/etf_data.csv`: ETF data file (may be generated or updated by the scripts).

## Limitations & Disclaimer
- This is a prototype and not intended for production or real investment decisions.
- Data sources and logic may be incomplete or contain errors.
- No warranty or support is provided.

---

*Developed as part of the Charles Schwab interview process by Armin Ahmadi.* 