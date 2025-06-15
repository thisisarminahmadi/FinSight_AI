"""
ETF Portfolio Recommendation System
==================================

This system provides AI-powered ETF portfolio recommendations tailored to user preferences,
leveraging natural language processing and robust financial data analysis.

Key Features
------------
• Natural Language Understanding: Extracts investment themes, budget, and risk tolerance from user queries.
• Portfolio-Level Theme Coverage: Ensures the recommended portfolio covers all requested themes, not just individual ETFs.
• Smart ETF Selection: Selects top ETFs for each theme, then fills remaining slots with best-scoring options, avoiding duplicates.
• Markdown Table Output: Presents recommendations in a clean, readable Markdown table for optimal UI display.
• Portfolio Metrics: Calculates and displays average return, expense ratio, volatility, and sector concentration.
• LLM Integration: Uses OpenAI's GPT models for intent extraction and concise, formatted portfolio explanations.
• User Experience: Provides warnings for negative performance, clear error messages, and actionable suggestions.
• Live Data: Uses Yahoo Finance API for real-time ETF data and performance metrics.

Usage
-----
1. Set up environment variables:
   - OPENAI_API_KEY: Your OpenAI API key
   - OPENAI_MODEL: GPT model to use (default: gpt-4.1)
   - MAX_ETFS: Maximum ETFs per portfolio (default: 4)
   - MIN_ALLOC: Minimum allocation per ETF in USD (default: 1000)

2. Run the script to launch the Gradio interface.
3. Enter queries like:
   - "I want to invest $15,000 in clean energy and AI, medium risk"
   - "Can you suggest a low-risk portfolio for $10,000 in healthcare?"

Constraints
-----------
• Minimum investment: $1,000 per ETF
• Maximum investment: $1,000,000 total
• Portfolio size: 1-4 ETFs
• Risk levels: low, medium, high

Author: Armin Ahmadi
Date: 2025-06-12
"""

# ============================
# Imports & Configuration
# ============================

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
import yfinance as yf
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.callbacks.manager import get_openai_callback
from datetime import datetime

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Environment ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set. Add it to your .env file or env vars.")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1")

# ---------- Data Storage ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
ETF_DATA_FILE = DATA_DIR / "etf_data.csv"
ETF_CACHE_FILE = DATA_DIR / "etf_cache.json"
CACHE_DURATION = 7200  # 2 hours in seconds

# ============================
# Risk Model & ETF Scoring
# ============================

RISK_THRESHOLDS = {
    "low": (0.0, 15.0),     # target volatility in %
    "medium": (15.0, 25.0),
    "high": (25.0, 100.0),
}

def calculate_risk_score(volatility: float, risk_level: str) -> float:
    """Calculate risk alignment score (0-1, higher is better)."""
    low, high = RISK_THRESHOLDS.get(risk_level, (0, 100))
    target = (low + high) / 2
    max_deviation = (high - low) / 2
    return 1 - min(abs(volatility - target) / max_deviation, 1)

def calculate_performance_score(row: pd.Series) -> float:
    """Calculate normalized performance score using multiple timeframes."""
    weights = {
        'performance_3m': 0.3,
        'performance_ytd': 0.3,
        'performance_52wk': 0.4
    }
    
    scores = {
        'performance_3m': min(max((row.performance_3m + 50) / 100, 0), 1),
        'performance_ytd': min(max((row.performance_ytd + 50) / 100, 0), 1),
        'performance_52wk': min(max((row.performance_52wk + 50) / 100, 0), 1)
    }
    
    return sum(score * weights[metric] for metric, score in scores.items())

def calculate_technical_score(row: pd.Series) -> float:
    """Calculate technical analysis score using price and moving averages."""
    scores = []
    
    # Price relative to moving averages
    if not pd.isna(row.price) and not pd.isna(row.ma50) and not pd.isna(row.ma200):
        ma50_score = 1 if row.price > row.ma50 else 0.5
        ma200_score = 1 if row.price > row.ma200 else 0.5
        scores.extend([ma50_score, ma200_score])
    
    # Volume trend
    if not pd.isna(row.volume):
        volume_score = min(row.volume / 1e6, 1)  # Normalize to 1M shares
        scores.append(volume_score)
    
    # 52-week range position
    if not pd.isna(row.week_52_high) and not pd.isna(row.week_52_low) and not pd.isna(row.price):
        range_score = (row.price - row.week_52_low) / (row.week_52_high - row.week_52_low)
        scores.append(range_score)
    
    return sum(scores) / len(scores) if scores else 0.5

def calculate_quality_score(row: pd.Series) -> float:
    """Calculate quality score based on various metrics."""
    scores = []
    
    # Expense ratio (lower is better)
    if not pd.isna(row.expense_ratio):
        expense_score = 1 - min(row.expense_ratio / 2, 1)  # 2% is max
        scores.append(expense_score)
    
    # Beta (closer to 1 is better)
    if not pd.isna(row.beta):
        beta_score = 1 - min(abs(row.beta - 1), 1)
        scores.append(beta_score)
    
    # PE ratio (lower is better, but not too low)
    if not pd.isna(row.pe_ratio) and row.pe_ratio > 0:
        pe_score = 1 - min(row.pe_ratio / 50, 1)  # 50 is max
        scores.append(pe_score)
    
    # Dividend yield (higher is better)
    if not pd.isna(row.dividend_yield):
        div_score = min(row.dividend_yield / 5, 1)  # 5% is max
        scores.append(div_score)
    
    # Market cap (larger is better)
    if not pd.isna(row.market_cap):
        cap_score = min(np.log10(row.market_cap + 1) / 6, 1)  # $1B ≈ 0.5, $100B ≈ 1.0
        scores.append(cap_score)
    
    return sum(scores) / len(scores) if scores else 0.5

def calculate_momentum_score(row: pd.Series) -> float:
    """Calculate momentum score based on recent price changes."""
    scores = []
    
    # Recent price change
    if not pd.isna(row.change_percent):
        change_score = min(max((row.change_percent + 10) / 20, 0), 1)  # -10% to +10% range
        scores.append(change_score)
    
    # 3-month performance
    if not pd.isna(row.performance_3m):
        perf_score = min(max((row.performance_3m + 20) / 40, 0), 1)  # -20% to +20% range
        scores.append(perf_score)
    
    return sum(scores) / len(scores) if scores else 0.5

def score_etf(row: pd.Series, risk_level: str) -> float:
    """Enhanced ETF scoring function with normalized metrics."""
    # Calculate component scores
    perf_score = calculate_performance_score(row)
    tech_score = calculate_technical_score(row)
    quality_score = calculate_quality_score(row)
    momentum_score = calculate_momentum_score(row)
    risk_score = calculate_risk_score(row.volatility, risk_level)
    
    # Weighted scoring
    weights = {
        'performance': 0.25,
        'technical': 0.20,
        'quality': 0.20,
        'momentum': 0.15,
        'risk': 0.20
    }
    
    return (
        perf_score * weights['performance'] +
        tech_score * weights['technical'] +
        quality_score * weights['quality'] +
        momentum_score * weights['momentum'] +
        risk_score * weights['risk']
    )

# ============================
# Data Management
# ============================

@dataclass
class YahooFinanceData:
    """Handles live ETF data fetching from Yahoo Finance with caching."""
    
    def __init__(self, cache_duration=CACHE_DURATION):
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = cache_duration
        self.df = self._load_or_fetch_data()
    
    def _get_all_etfs(self) -> List[str]:
        """Get list of all ETFs from the predefined list."""
        return [
            "TSLL", "SOXL", "SQQQ", "SOXS", "TSLZ", "TQQQ", "SPXS", "SPY", "FAZ", "MSTU",
            "QQQ", "XLF", "XLE", "IWM", "IBIT", "UVXY", "AMDL", "MSTZ", "LQD", "TLT",
            "TSLQ", "FXI", "UVIX", "HYG", "EEM", "USO", "NVDQ", "TZA", "EWZ", "SCHD",
            "TNA", "ETHA", "SPXU", "GLD", "TSLS", "GDX", "SLV", "BITO", "EFA", "SPDN",
            "SDS", "LABD", "SCO", "KRE", "KWEB", "NVDL", "NVDX", "XLI", "UCO", "XLU",
            "NVDD", "ULTY", "XLV", "RSP", "TMF", "IAU", "XBI", "QID", "XLP", "GDXD",
            "VXX", "SCHX", "ARKK", "SMH", "IEMG", "PLTD", "FNGB", "IEFA", "BITO", "SPLG",
            "VEA", "SCHG", "BITX", "SVIX", "MSTY", "RWM", "PSQ", "UPRO", "VOO", "EMB",
            "XOP", "SCHA", "SGOV", "GOVT", "SH", "NVD", "JEPQ", "SOXX", "TSLY", "TSLG",
            "TSLT", "CONY", "PLTU", "JDST", "VWO", "UNG", "IEF", "VCIT", "ETHU", "EWG",
            "VPU", "VPLS", "VPL", "VOX", "VOTE", "VOT", "VOOV", "VOOG", "VOO", "VONV",
            "VONG", "VONE", "VOE", "VO", "VNQI", "VNQ", "VNM", "VNLA", "VMBS", "VLUE",
            "VIXY", "VIXM", "VIOO", "VIGI", "VIG", "VHT", "VGUS", "VGT", "VGSH", "VGMS",
            "VGLT", "VGK", "VGIT", "VFLO", "VFH", "VEU", "VEA", "VDE", "VDC", "VCSH",
            "VCRM", "VCRB", "VCR", "VCLT", "VCIT", "VBR", "VBK", "VBIL", "VB", "UYLD",
            "UWM", "UVXY", "UVIX", "UUP", "UTWO", "UTSL", "UTES", "UTEN", "USXF", "USRT",
            "USOY", "USOI", "USO", "USMV", "USMC", "USIG", "USHY", "USFR", "USD", "URTY",
            "URTH", "URNM", "URNJ", "URAA", "URA", "UPRO", "UNG", "ULTY", "ULST", "ULE",
            "UITB", "UGL", "UDOW", "UDN", "UCON", "UCO", "UBRL", "UBOT", "UBND", "UAE",
            "TZA", "TYD", "TYA", "TWM", "TUSI", "TUR", "TUA", "TSYY", "TSPA", "TSMY"
        ]
    
    def _load_cache(self) -> None:
        """Load cache from file if it exists and is not expired."""
        try:
            if ETF_CACHE_FILE.exists():
                with open(ETF_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    self.cache = cache_data.get('cache', {})
                    self.cache_time = cache_data.get('cache_time', {})
                    
                    # Remove expired cache entries
                    current_time = time.time()
                    expired = [k for k, v in self.cache_time.items() 
                             if current_time - v > self.cache_duration]
                    for k in expired:
                        del self.cache[k]
                        del self.cache_time[k]
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
            self.cache = {}
            self.cache_time = {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(ETF_CACHE_FILE, 'w') as f:
                json.dump({
                    'cache': self.cache,
                    'cache_time': self.cache_time
                }, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def _load_or_fetch_data(self) -> pd.DataFrame:
        """Load data from CSV or fetch new data if needed."""
        if ETF_DATA_FILE.exists():
            df = pd.read_csv(ETF_DATA_FILE)
            file_age = time.time() - ETF_DATA_FILE.stat().st_mtime
            if file_age < self.cache_duration:
                logger.info("Using cached ETF data")
                return df
        
        logger.info("Fetching fresh ETF data")
        df = self._fetch_all_etf_data()
        df.to_csv(ETF_DATA_FILE, index=False)
        return df
    
    def _fetch_etf_data(self, ticker: str) -> Dict:
        """Fetch data for a single ETF with caching."""
        current_time = time.time()
        
        # Check cache first
        if ticker in self.cache and current_time - self.cache_time[ticker] < self.cache_duration:
            return self.cache[ticker]
        
        try:
            etf = yf.Ticker(ticker)
            info = etf.info
            
            # Get historical data for various metrics
            hist = etf.history(period="1y")  # 1 year of data
            hist_3m = etf.history(period="3mo")  # 3 months of data
            hist_ytd = etf.history(start=f"{datetime.now().year}-01-01")  # Year to date
            
            # Calculate performance metrics
            performance_52wk = 0
            performance_3m = 0
            performance_ytd = 0
            volatility = 0
            
            if len(hist) > 0:
                performance_52wk = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                volatility = hist['Close'].pct_change().std() * 100
            
            if len(hist_3m) > 0:
                performance_3m = ((hist_3m['Close'].iloc[-1] / hist_3m['Close'].iloc[0]) - 1) * 100
            
            if len(hist_ytd) > 0:
                performance_ytd = ((hist_ytd['Close'].iloc[-1] / hist_ytd['Close'].iloc[0]) - 1) * 100
            
            # Get expense ratio from multiple possible fields
            expense_ratio = 0
            for field in ['annualReportExpenseRatio', 'fundInceptionDate', 'totalAssets']:
                if field in info and info[field]:
                    if field == 'annualReportExpenseRatio':
                        expense_ratio = info[field] * 100
                    break
            
            # Initialize ETF data with calculated values
            etf_data = {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0),
                'volume': info.get('regularMarketVolume', 0),
                'ma50': info.get('fiftyDayAverage', 0),
                'ma200': info.get('twoHundredDayAverage', 0),
                'performance_3m': performance_3m,
                'performance_ytd': performance_ytd,
                'performance_52wk': performance_52wk,
                'week_52_high': info.get('fiftyTwoWeekHigh', 0),
                'week_52_low': info.get('fiftyTwoWeekLow', 0),
                'expense_ratio': expense_ratio,
                'sector': self._determine_sector(info),
                'theme_tags': self._get_theme_tags(ticker, info),
                'risk_level': 'medium',  # Will be updated based on volatility
                'aum': info.get('totalAssets', 0) / 1e6,  # Convert to millions
                'volatility': volatility,
                'dividend_yield': info.get('dividendYield', 0) * 100,
                'market_cap': info.get('totalAssets', 0) / 1e6,  # Use AUM instead of market cap
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0),  # Default to 1.0 if missing
                'quality_score': 0  # Will be calculated
            }
            
            # Determine risk level based on volatility
            etf_data['risk_level'] = self._determine_risk_level(etf_data['volatility'])
            
            # Calculate quality score
            etf_data['quality_score'] = calculate_quality_score(pd.Series(etf_data))
            
            # Update cache
            self.cache[ticker] = etf_data
            self.cache_time[ticker] = current_time
            
            return etf_data
            
        except Exception as e:
            logger.warning(f"Error fetching data for {ticker}: {e}")
            return None
    
    def _fetch_all_etf_data(self) -> pd.DataFrame:
        """Fetch data for all ETFs and create a DataFrame."""
        etf_data = []
        tickers = self._get_all_etfs()
        total_etfs = len(tickers)
        
        logger.info(f"Starting to fetch data for {total_etfs} ETFs...")
        
        for i, ticker in enumerate(tickers, 1):
            data = self._fetch_etf_data(ticker)
            if data:
                etf_data.append(data)
            
            # Log progress every 50 ETFs
            if i % 50 == 0:
                logger.info(f"Processed {i}/{total_etfs} ETFs...")
        
        # Create DataFrame and clean data
        df = pd.DataFrame(etf_data)
        df = self._clean_data(df)
        
        logger.info(f"Successfully fetched data for {len(df)} out of {total_etfs} ETFs")
        return df
    
    def _determine_sector(self, info: Dict) -> str:
        """Determine sector based on ETF information."""
        # Try to get sector from info
        sector = info.get('sector', '')
        if sector:
            return sector
        
        # If no sector, try to determine from name and description
        name = info.get('longName', '').lower()
        description = info.get('longBusinessSummary', '').lower()
        
        sector_keywords = {
            'Technology': ['tech', 'technology', 'software', 'hardware', 'semiconductor', 'digital', 'internet'],
            'Energy': ['energy', 'oil', 'gas', 'solar', 'wind', 'renewable', 'clean'],
            'Healthcare': ['health', 'medical', 'biotech', 'pharma', 'healthcare'],
            'Financial': ['financial', 'banking', 'insurance', 'fintech'],
            'Real Estate': ['real estate', 'reit', 'property'],
            'Consumer': ['consumer', 'retail', 'discretionary', 'staples'],
            'Industrial': ['industrial', 'manufacturing', 'materials'],
            'Utilities': ['utilities', 'electric', 'water', 'gas'],
            'Materials': ['materials', 'mining', 'metals', 'chemicals'],
            'Communication': ['communication', 'telecom', 'media', 'entertainment']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in name or keyword in description for keyword in keywords):
                return sector
        
        return 'Diversified'  # Default if no sector can be determined
    
    def _get_theme_tags(self, ticker: str, info: Dict) -> str:
        """Extract theme tags from ETF information."""
        themes = []
        
        # Add sector as a theme
        sector = self._determine_sector(info)
        if sector:
            themes.append(sector.lower())
        
        # Add specific themes based on name and description
        name = info.get('longName', '').lower()
        description = info.get('longBusinessSummary', '').lower()
        
        # Enhanced theme keywords
        theme_keywords = {
            'clean energy': ['clean', 'energy', 'solar', 'wind', 'renewable', 'green', 'sustainable', 'carbon', 'climate', 'environmental'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'robotics', 'automation', 'tech', 'technology', 'digital', 'smart', 'innovation'],
            'technology': ['tech', 'technology', 'software', 'hardware', 'semiconductor', 'digital', 'internet', 'cloud', 'cyber', 'data', 'computing'],
            'healthcare': ['health', 'medical', 'biotech', 'pharma', 'healthcare', 'life sciences', 'genomics', 'therapeutics'],
            'esg': ['esg', 'sustainable', 'green', 'social', 'governance', 'ethical', 'responsible', 'impact'],
            'financial': ['financial', 'banking', 'insurance', 'fintech', 'finance', 'bank', 'credit', 'lending'],
            'real estate': ['real estate', 'reit', 'property', 'housing', 'commercial', 'residential'],
            'consumer': ['consumer', 'retail', 'discretionary', 'staples', 'goods', 'services', 'brands'],
            'industrial': ['industrial', 'manufacturing', 'materials', 'engineering', 'construction', 'machinery'],
            'utilities': ['utilities', 'electric', 'water', 'gas', 'power', 'energy', 'infrastructure'],
            'leveraged': ['leveraged', '2x', '3x', 'bull', 'bear', 'inverse', 'ultra', 'pro'],
            'inverse': ['inverse', 'short', 'bear', 'negative', 'opposite'],
            'volatility': ['volatility', 'vix', 'fear', 'greed', 'market sentiment'],
            'dividend': ['dividend', 'income', 'yield', 'payout', 'distribution'],
            'growth': ['growth', 'momentum', 'innovation', 'disruptive', 'emerging'],
            'value': ['value', 'fundamental', 'quality', 'dividend', 'income']
        }
        
        # Check name and description for themes
        for theme, keywords in theme_keywords.items():
            if any(keyword in name or keyword in description for keyword in keywords):
                themes.append(theme)
        
        # Add additional themes based on ETF name patterns
        name_patterns = {
            'clean energy': ['solar', 'wind', 'clean', 'green', 'renewable', 'env', 'climate'],
            'ai': ['ai', 'robot', 'tech', 'digital', 'smart', 'ml', 'data'],
            'esg': ['esg', 'sustainable', 'green', 'social', 'governance'],
            'tech': ['tech', 'digital', 'internet', 'cloud', 'cyber'],
            'leveraged': ['2x', '3x', 'ultra', 'pro', 'bull', 'bear'],
            'inverse': ['inverse', 'short', 'bear', 'opposite'],
            'volatility': ['vix', 'vol', 'fear', 'greed']
        }
        
        for theme, patterns in name_patterns.items():
            if any(pattern in name.lower() for pattern in patterns):
                themes.append(theme)
        
        # Add leveraged/inverse themes based on name
        if any(x in name.lower() for x in ['2x', '3x', 'ultra', 'pro']):
            themes.append('leveraged')
        if any(x in name.lower() for x in ['inverse', 'short', 'bear']):
            themes.append('inverse')
        
        return ','.join(set(themes))
    
    def _determine_risk_level(self, volatility: float) -> str:
        """Determine risk level based on volatility."""
        if pd.isna(volatility):
            return 'medium'
        if volatility < 15:
            return 'low'
        elif volatility < 25:
            return 'medium'
        else:
            return 'high'
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the ETF data."""
        if df.empty:
            logger.warning("No ETF data available")
            return pd.DataFrame(columns=[
                'ticker', 'name', 'description', 'expense_ratio', 'sector',
                'theme_tags', 'risk_level', 'aum', 'performance_52wk',
                'performance_3m', 'performance_ytd', 'volatility', 'volume',
                'ma50', 'ma200', 'week_52_high', 'week_52_low',
                'dividend_yield', 'price', 'market_cap', 'pe_ratio', 'beta'
            ])
        
        # Fill missing values and ensure proper types
        df['theme_tags'] = df['theme_tags'].fillna('').astype(str)
        df['sector'] = df['sector'].fillna('Unknown').astype(str)
        df['risk_level'] = df['risk_level'].fillna('medium').astype(str)
        
        # Clean numeric fields
        numeric_cols = [
            'expense_ratio', 'performance_52wk', 'performance_3m', 'performance_ytd',
            'volatility', 'dividend_yield', 'aum', 'volume', 'ma50', 'ma200',
            'week_52_high', 'week_52_low', 'price', 'market_cap', 'pe_ratio', 'beta'
        ]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
        
        # Remove rows with missing critical metrics
        crit_cols = ["performance_52wk", "aum", "volatility"]
        pre_drop = len(df)
        df = df.dropna(subset=crit_cols)
        logger.info(
            "Dropped %d malformed rows; %d remaining.",
            pre_drop - len(df), len(df)
        )
        
        return df
    
    def get_sector_overlap(self, etfs: List[str]) -> float:
        """Calculate sector overlap between selected ETFs."""
        sectors = self.df[self.df['ticker'].isin(etfs)]['sector'].value_counts()
        return (sectors / len(etfs)).max()  # Returns highest sector concentration

# Initialize Yahoo Finance data
try:
    etf_data = YahooFinanceData()
    logger.info("Successfully loaded live ETF data from Yahoo Finance")
except Exception as e:
    logger.error("Failed to load ETF data: %s", e)
    raise

# ============================
# LLM: Intent Extraction
# ============================

response_schemas = [
    ResponseSchema(name="themes", description="List of investment themes"),
    ResponseSchema(name="budget", description="Budget in USD (number)"),
    ResponseSchema(name="risk_level", description="Risk tolerance: low, medium, or high"),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

intent_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Extract the user's investment preferences from the message.\n"
        "Return JSON with keys: themes (array of strings), budget (number), risk_level (string).\n\n"
        "User message: \n{user_input}\n\n{format_instructions}"
    ),
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm_intent = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0)
extract_intent_chain = intent_prompt | llm_intent | parser

# ============================
# Portfolio Construction
# ============================

def allocate_portfolio(budget: float, etfs: pd.DataFrame, max_etfs: int = 4) -> pd.DataFrame:
    """Enhanced portfolio allocation with diversification checks."""
    pick = etfs.head(max_etfs).copy()
    
    # Ensure minimum budget for max_etfs
    min_total_budget = 1000 * max_etfs  # $1k per ETF
    if budget < min_total_budget:
        max_etfs = max(1, budget // 1000)  # Reduce max_etfs to fit budget
        pick = pick.head(max_etfs)
        logger.warning(
            "Budget $%d too low for %d ETFs. Reducing to %d ETFs.",
            budget, max_etfs, len(pick)
        )
    
    # Pre-filter by sector to ensure diversification
    selected_sectors = set()
    diversified_pick = []
    
    for _, row in pick.iterrows():
        if row['sector'] not in selected_sectors or len(diversified_pick) < 2:
            diversified_pick.append(row)
            selected_sectors.add(row['sector'])
    
    pick = pd.DataFrame(diversified_pick)
    
    # Calculate initial weights based on scores
    total_score = pick["score"].sum()
    if total_score == 0:
        weights = [1 / len(pick)] * len(pick)
    else:
        weights = pick["score"] / total_score
    
    # Calculate allocations
    pick["allocation"] = (weights * budget).round(-2).astype(int)
    
    # Ensure minimum allocation of $1,000 per ETF
    min_allocation = 1000
    while any(pick["allocation"] < min_allocation) and len(pick) > 1:
        # Remove ETF with lowest allocation
        pick = pick[pick["allocation"] >= min_allocation]
        # Recalculate weights and allocations
        weights = pick["score"] / pick["score"].sum()
        pick["allocation"] = (weights * budget).round(-2).astype(int)
    
    return pick

def build_explanation_prompt(picks: pd.DataFrame) -> str:
    """Build a prompt for the LLM to explain the portfolio."""
    # Calculate portfolio metrics
    total_allocation = picks["allocation"].sum()
    portfolio_metrics = {
        'avg_1yr_return': (picks['performance_52wk'] * picks['allocation']).sum() / total_allocation,
        'avg_expense_ratio': (picks['expense_ratio'] * picks['allocation']).sum() / total_allocation,
        'portfolio_volatility': (picks['volatility'] * picks['allocation']).sum() / total_allocation,
        'portfolio_quality': (picks['quality_score'] * picks['allocation']).sum() / total_allocation,
        'sector_concentration': picks['sector'].value_counts().max() / len(picks)
    }
    
    # Build the prompt with proper markdown formatting
    prompt = f"""Please analyze this ETF portfolio and provide a concise explanation of its composition and characteristics.

Portfolio Metrics:
• Average 1-Year Return: {portfolio_metrics['avg_1yr_return']:.1f}%
• Average Expense Ratio: {portfolio_metrics['avg_expense_ratio']:.2f}%
• Portfolio Volatility: {portfolio_metrics['portfolio_volatility']:.1f}%
• Portfolio Quality Score: {portfolio_metrics['portfolio_quality']:.2f}
• Sector Concentration: {portfolio_metrics['sector_concentration']:.1%}

ETF Selections:
| Ticker | Allocation | 1Y Return | 3M Return | YTD | Vol | MA50/200 | Volume | Quality | Sector |
|--------|------------|-----------|-----------|-----|-----|----------|---------|---------|---------|
"""
    
    # Add each ETF to the table
    for _, etf in picks.iterrows():
        ma_status = "↑" if etf.price > etf.ma50 and etf.price > etf.ma200 else "↓"
        volume_str = f"{etf.volume/1e6:.1f}M"
        allocation_pct = (etf.allocation / total_allocation) * 100
        prompt += f"| {etf.ticker} | ${etf.allocation:,.0f} ({allocation_pct:.1f}%) | {etf.performance_52wk:.1f}% | {etf.performance_3m:.1f}% | {etf.performance_ytd:.1f}% | {etf.volatility:.1f}% | {ma_status} | {volume_str} | {etf.quality_score:.2f} | {etf.sector} |\n"
    
    # Add few-shot example
    example = """
Example Response Format:

| Ticker | Allocation | 1Y Return | 3M Return | YTD | Vol | MA50/200 | Volume | Quality | Sector |
|--------|------------|-----------|-----------|-----|-----|----------|---------|---------|---------|
| BOTZ   | $50,000 (50.0%) | 15.20% | 12.40% | 18.00% | 18.00% | ↑ | 2.1M | 0.75 | Technology |
| ARKK   | $50,000 (50.0%) | 12.40% | 10.20% | 25.00% | 25.00% | ↑ | 5.3M | 0.65 | Technology |

Portfolio Metrics:
• Average 1-Year Return: 13.80%
• Average Expense Ratio: 0.72%
• Portfolio Volatility: 21.50%
• Portfolio Quality Score: 0.70
• Sector Concentration: 100.0%

This portfolio offers strong exposure to AI and innovation with a balanced allocation between robotics (BOTZ) and disruptive tech (ARKK). The 21.5% volatility is appropriate for medium risk tolerance. Consider adding a small allocation to a low-volatility tech ETF to reduce overall risk.
"""
    
    prompt += f"\n{example}\n"
    prompt += "Please provide a brief analysis of this portfolio following the exact format above. Keep the analysis under 300 words and maintain the Markdown table format exactly as shown."
    
    # Add warning if all ETFs have negative returns
    if all(picks['performance_52wk'] < 0):
        prompt += "\n\nNote: All selected ETFs have negative 1-year returns. Consider diversifying if concerned about short-term performance."
    
    return prompt

llm_explainer = ChatOpenAI(model_name=MODEL_NAME, temperature=0.1)

# ============================
# Chatbot Logic
# ============================

def validate_user_input(intent: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate extracted user intent."""
    if not intent["themes"]:
        return False, "Please specify at least one investment theme."
    
    try:
        budget = float(intent["budget"])
        max_etfs = int(os.getenv("MAX_ETFS", "4"))
        min_alloc = int(os.getenv("MIN_ALLOC", "1000"))
        
        if budget < min_alloc:
            return False, f"Minimum investment amount is ${min_alloc:,}."
        if budget < min_alloc * max_etfs:
            return False, (
                f"Budget too low for {max_etfs} ETFs. "
                f"Need at least ${min_alloc * max_etfs:,} "
                f"(${min_alloc:,} per ETF)."
            )
        if budget > 1000000:
            return False, "Maximum investment amount is $1,000,000."
    except (ValueError, TypeError):
        return False, "Please specify a valid budget amount."
    
    if intent["risk_level"].lower() not in RISK_THRESHOLDS:
        return False, "Please specify risk level as low, medium, or high."
    
    return True, ""

def chatbot_handler(message: str, history: List[List[str]]) -> str:
    """Enhanced chatbot handler with better error handling and user experience."""
    try:
        # ---- Intent extraction ----
        with get_openai_callback() as cb:
            intent = extract_intent_chain.invoke({"user_input": message})
            logger.info("Intent extraction cost: %s", cb)
        
        # ---- Input validation ----
        is_valid, error_msg = validate_user_input(intent)
        if not is_valid:
            return f"❌ {error_msg}"
        
        themes: List[str] = intent["themes"]
        budget: float = float(intent["budget"])
        risk_level: str = intent["risk_level"].lower()
        
        # ---- ETF retrieval ----
        candidate_etfs = filter_etfs(themes, risk_level)
        if candidate_etfs.empty:
            return (
                "😕 I couldn't find ETFs matching those themes. "
                "Try different keywords or a different risk level. "
                "For example, you could try:\n"
                "• Different theme keywords\n"
                "• A broader risk level\n"
                "• A different investment amount"
            )
        
        # ---- Portfolio construction ----
        portfolio = allocate_portfolio(budget, candidate_etfs)
        
        # ---- LLM explanation ----
        prompt = build_explanation_prompt(portfolio)
        
        with get_openai_callback() as cb:
            reply = llm_explainer.invoke(prompt).content.strip()
            logger.info("Portfolio explanation cost: %s", cb)
        
        return reply

    except Exception as exc:
        logger.exception("Chatbot error: %s", exc)
        return (
            "⚠️ I encountered an error while processing your request. "
            "Please try again or rephrase your question. "
            "For example:\n"
            "• 'I want to invest $15,000 in clean energy and AI, medium risk'\n"
            "• 'Can you suggest a low-risk portfolio for $10,000 in healthcare?'"
        )

# ============================
# Gradio UI
# ============================

def build_ui() -> gr.Blocks:
    """Enhanced UI with better user guidance."""
    with gr.Blocks(title="FinSight AI") as demo:
        gr.Markdown("""
        # FinSight AI
        
        Get personalized ETF portfolio recommendations based on your investment goals.
        
        ## How to use
        1. Tell me your investment amount
        2. Specify your themes of interest
        3. Mention your risk tolerance (low/medium/high)
        
        ## Example queries
        • "I want to invest $15,000 in clean energy and AI, medium risk"
        • "Can you suggest a low-risk portfolio for $10,000 in healthcare?"
        • "I have $25,000 for tech and renewable energy, high risk"
        
        ## Notes
        • Minimum investment: $1,000
        • Maximum investment: $1,000,000
        • Portfolio size: 2-4 ETFs
        • Minimum allocation per ETF: $1,000
        """)
        
        chat = gr.ChatInterface(
            fn=chatbot_handler,
            examples=[
                "I want to invest $15,000 in clean energy and AI, medium risk",
                "Can you suggest a low-risk portfolio for $10,000 in healthcare?",
                "I have $25,000 for tech and renewable energy, high risk"
            ]
        )
        
        gr.Markdown("""
        ---
        *Powered by OpenAI + LangChain + Gradio*
        
        This is a demonstration project and not financial advice. 
        Always consult with a financial advisor before making investment decisions.
        """)
    
    return demo

def filter_etfs(themes: List[str], risk_level: str, top_n: int = 10) -> pd.DataFrame:
    """Return top-N ETFs ensuring portfolio covers all requested themes."""
    cleaned_themes = [t.lower().replace(" ", "") for t in themes]
    selected = []
    used_tickers = set()
    
    # Helper function to safely check theme in tags
    def theme_in_tags(tags: str, theme: str) -> bool:
        if pd.isna(tags) or not isinstance(tags, str):
            return False
        return theme in tags.lower().split(",")
    
    # 1. For each theme, pick the best ETF for that theme (with risk match)
    for theme in cleaned_themes:
        theme_mask = etf_data.df["theme_tags"].apply(lambda tags: theme_in_tags(tags, theme))
        risk_mask = etf_data.df["risk_level"].str.lower().eq(risk_level.lower())
        subset = etf_data.df[theme_mask & risk_mask & (~etf_data.df['ticker'].isin(used_tickers))].copy()
        
        if not subset.empty:
            subset["score"] = subset.apply(score_etf, axis=1, risk_level=risk_level.lower())
            best = subset.sort_values("score", ascending=False).iloc[0]
            selected.append(best)
            used_tickers.add(best.ticker)
            logger.info(f"Selected {best.ticker} for theme {theme}")
    
    # 2. Fill remaining slots with best-scoring ETFs from any theme (no duplicates)
    remaining = top_n - len(selected)
    if remaining > 0:
        # Allow any ETF matching any requested theme, not already selected
        theme_mask = etf_data.df["theme_tags"].apply(
            lambda tags: any(theme_in_tags(tags, theme) for theme in cleaned_themes)
        )
        risk_mask = etf_data.df["risk_level"].str.lower().eq(risk_level.lower())
        subset = etf_data.df[theme_mask & risk_mask & (~etf_data.df['ticker'].isin(used_tickers))].copy()
        
        if not subset.empty:
            subset["score"] = subset.apply(score_etf, axis=1, risk_level=risk_level.lower())
            fill = subset.sort_values("score", ascending=False).head(remaining)
            selected.extend([row for _, row in fill.iterrows()])
            logger.info(f"Added {len(fill)} additional ETFs to fill remaining slots")
    
    # 3. If still not enough, fill with any ETF (ignore risk), still no duplicates
    remaining = top_n - len(selected)
    if remaining > 0:
        theme_mask = etf_data.df["theme_tags"].apply(
            lambda tags: any(theme_in_tags(tags, theme) for theme in cleaned_themes)
        )
        subset = etf_data.df[theme_mask & (~etf_data.df['ticker'].isin(used_tickers))].copy()
        
        if not subset.empty:
            subset["score"] = subset.apply(score_etf, axis=1, risk_level=risk_level.lower())
            fill = subset.sort_values("score", ascending=False).head(remaining)
            selected.extend([row for _, row in fill.iterrows()])
            logger.info(f"Added {len(fill)} ETFs ignoring risk level")
    
    # 4. Return as DataFrame
    if selected:
        result = pd.DataFrame(selected).head(top_n)
        logger.info(f"Selected ETFs: {', '.join(result['ticker'].tolist())}")
        return result
    else:
        logger.warning("No ETFs found matching the criteria")
        return pd.DataFrame(columns=etf_data.df.columns.tolist() + ["score"])

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
