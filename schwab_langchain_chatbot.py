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

Usage
-----
1. Set up environment variables:
   - OPENAI_API_KEY: Your OpenAI API key
   - ETF_CSV: Path to your ETF data CSV file
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
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.callbacks.manager import get_openai_callback

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

ETF_CSV = Path(os.getenv("ETF_CSV", "etf_data.csv")).expanduser()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1")

# ============================
# Data Loading & Cleaning
# ============================

@dataclass
class ETFData:
    """Validated ETF dataset with helper methods."""
    df: pd.DataFrame
    
    def __post_init__(self):
        self._clean_data()  # Clean first
        self._validate_data()  # Then validate
    
    def _clean_data(self):
        """Clean and normalize the data."""
        # Normalize column names
        self.df.columns = self.df.columns.str.lower().str.strip()
        
        # Clean text fields
        self.df["theme_tags"] = (
            self.df["theme_tags"]
            .fillna("")
            .str.lower()
            .str.replace(r"\s+", "", regex=True)
        )
        
        # Map risk level text to numeric values
        RISK_LEVEL_MAP = {
            'low': 10.0,
            'medium': 20.0,
            'high': 30.0,
            'very high': 40.0,
            'extremely high': 50.0
        }
        
        # Convert risk levels to numeric values
        self.df['volatility'] = (
            self.df['volatility']
            .fillna("")
            .str.lower()
            .str.strip()
            .map(RISK_LEVEL_MAP)
            .fillna(pd.NA)
        )
        
        # Clean numeric fields with specific patterns
        CLEAN_MAP = {
            "expense_ratio": r"[%]",  # Remove % symbol
            "performance_1yr": r"[%]",  # Remove % symbol
            "dividend_yield": r"[%]",  # Remove % symbol
            "aum": r"[\$,mMbB,]",  # Remove currency and unit symbols
        }
        
        for col, pat in CLEAN_MAP.items():
            # First pass: remove symbols and normalize
            self.df[col] = (
                self.df[col]
                .fillna("")  # Convert NA to empty string first
                .astype(str)
                .str.replace(pat, "", regex=True)
                .str.strip()
            )
            
            # Log any non-numeric values
            bad_vals = self.df[~self.df[col].str.match(r"^-?\d+(\.\d+)?$")][col].unique()
            if len(bad_vals):
                logger.warning(
                    "Found non-numeric values in %s: %s. These will be converted to NaN.",
                    col, bad_vals
                )
            
            # Second pass: clean and convert to float
            self.df[col] = (
                self.df[col]
                .replace(r"[^0-9\.\-]", "", regex=True)  # Strip any remaining non-numeric chars
                .replace("", pd.NA)  # Convert empty strings to NA
                .replace("nan", pd.NA)  # Convert 'nan' strings to NA
                .replace("None", pd.NA)  # Convert 'None' strings to NA
            )
            
            # Convert to float, handling NA values
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            except Exception as e:
                logger.error(f"Error converting {col} to numeric: {e}")
                raise
        
        # Convert AUM to millions if needed (assuming values are in millions)
        if 'aum' in self.df.columns:
            self.df['aum'] = self.df['aum'].fillna(0)  # Fill NA with 0 for AUM
        
        # Remove rows with missing critical metrics
        crit_cols = ["performance_1yr", "aum", "volatility"]
        pre_drop = len(self.df)
        self.df = self.df.dropna(subset=crit_cols)
        logger.info(
            "Dropped %d malformed rows; %d remaining. "
            "Check warnings above for details about removed data.",
            pre_drop - len(self.df), len(self.df)
        )
        
        # Log sample of cleaned data
        logger.info("Sample of cleaned data:\n%s", self.df.head().to_string())
    
    def _validate_data(self):
        """Ensure required columns and data types are present."""
        required_cols = {
            'ticker': str,
            'name': str,
            'description': str,
            'expense_ratio': float,
            'sector': str,
            'theme_tags': str,
            'risk_level': str,
            'aum': float,
            'performance_1yr': float,
            'volatility': float,
            'dividend_yield': float
        }
        
        missing_cols = set(required_cols.keys()) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data types after cleaning
        for col, expected_type in required_cols.items():
            actual_type = self.df[col].dtype
            if col in ['ticker', 'name', 'description', 'sector', 'theme_tags', 'risk_level']:
                if not pd.api.types.is_string_dtype(actual_type):
                    raise ValueError(f"Column {col} should be string type, got {actual_type}")
            else:
                if not pd.api.types.is_numeric_dtype(actual_type):
                    raise ValueError(f"Column {col} should be numeric type, got {actual_type}")
    
    def get_sector_overlap(self, etfs: List[str]) -> float:
        """Calculate sector overlap between selected ETFs."""
        sectors = self.df[self.df['ticker'].isin(etfs)]['sector'].value_counts()
        return (sectors / len(etfs)).max()  # Returns highest sector concentration

# Load and validate ETF data
try:
    df_raw = pd.read_csv(ETF_CSV)
    etf_data = ETFData(df_raw)
    logger.info("Successfully loaded and validated ETF data")
except Exception as e:
    logger.error("Failed to load ETF data: %s", e)
    raise

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

def calculate_performance_score(performance: float) -> float:
    """Calculate normalized performance score (0-1, higher is better)."""
    # Normalize to 0-1 range, assuming typical ETF returns
    return min(max((performance + 50) / 100, 0), 1)

def calculate_aum_score(aum: float) -> float:
    """Calculate AUM score (0-1, higher is better)."""
    # Improved scaling: $1B ≈ 0.5, $100B ≈ 1.0
    return min(np.log10(aum + 1) / 6, 1)

def score_etf(row: pd.Series, risk_level: str) -> float:
    """Enhanced ETF scoring function with normalized metrics."""
    # Normalize all components to 0-1 scale
    perf_score = calculate_performance_score(row.performance_1yr)
    aum_score = calculate_aum_score(row.aum)
    risk_score = calculate_risk_score(row.volatility, risk_level)
    fee_penalty = row.expense_ratio / 100.0  # Already in percentage
    
    # Weighted scoring
    weights = {
        'performance': 0.35,
        'aum': 0.25,
        'risk': 0.30,
        'fee': 0.10
    }
    
    return (
        perf_score * weights['performance'] +
        aum_score * weights['aum'] +
        risk_score * weights['risk'] -
        fee_penalty * weights['fee']
    )

def filter_etfs(themes: List[str], risk_level: str, top_n: int = 10) -> pd.DataFrame:
    """Return top-N ETFs ensuring portfolio covers all requested themes."""
    cleaned_themes = [t.lower().replace(" ", "") for t in themes]
    selected = []
    used_tickers = set()
    # 1. For each theme, pick the best ETF for that theme (with risk match)
    for theme in cleaned_themes:
        theme_mask = etf_data.df["theme_tags"].apply(lambda tags: theme in tags.split(","))
        risk_mask = etf_data.df["risk_level"].str.lower().eq(risk_level.lower())
        subset = etf_data.df[theme_mask & risk_mask & (~etf_data.df['ticker'].isin(used_tickers))].copy()
        if not subset.empty:
            subset["score"] = subset.apply(score_etf, axis=1, risk_level=risk_level.lower())
            best = subset.sort_values("score", ascending=False).iloc[0]
            selected.append(best)
            used_tickers.add(best.ticker)
    # 2. Fill remaining slots with best-scoring ETFs from any theme (no duplicates)
    remaining = top_n - len(selected)
    if remaining > 0:
        # Allow any ETF matching any requested theme, not already selected
        theme_mask = etf_data.df["theme_tags"].apply(lambda tags: any(theme in tags.split(",") for theme in cleaned_themes))
        risk_mask = etf_data.df["risk_level"].str.lower().eq(risk_level.lower())
        subset = etf_data.df[theme_mask & risk_mask & (~etf_data.df['ticker'].isin(used_tickers))].copy()
        if not subset.empty:
            subset["score"] = subset.apply(score_etf, axis=1, risk_level=risk_level.lower())
            fill = subset.sort_values("score", ascending=False).head(remaining)
            selected.extend([row for _, row in fill.iterrows()])
    # 3. If still not enough, fill with any ETF (ignore risk), still no duplicates
    remaining = top_n - len(selected)
    if remaining > 0:
        theme_mask = etf_data.df["theme_tags"].apply(lambda tags: any(theme in tags.split(",") for theme in cleaned_themes))
        subset = etf_data.df[theme_mask & (~etf_data.df['ticker'].isin(used_tickers))].copy()
        if not subset.empty:
            subset["score"] = subset.apply(score_etf, axis=1, risk_level=risk_level.lower())
            fill = subset.sort_values("score", ascending=False).head(remaining)
            selected.extend([row for _, row in fill.iterrows()])
    # 4. Return as DataFrame
    if selected:
        return pd.DataFrame(selected).head(top_n)
    else:
        return pd.DataFrame(columns=etf_data.df.columns.tolist() + ["score"])

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

def build_explanation_prompt(user_prefs: Dict[str, Any], picks: pd.DataFrame) -> str:
    """Enhanced prompt for more detailed portfolio explanation with Markdown table."""
    etf_rows = []
    total_allocation = picks["allocation"].sum()
    # Markdown table header with Name as second column
    etf_rows.append("| Ticker | Name | Allocation | 1Y Return | Volatility | Expense Ratio | Dividend Yield | Description |")
    etf_rows.append("|--------|-------------------------------|------------|-----------|------------|---------------|---------------|-------------|")
    for _, row in picks.iterrows():
        allocation_pct = (row.allocation / total_allocation) * 100
        etf_rows.append(
            f"| {row.ticker} | {row.name} | ${row.allocation:,.0f} ({allocation_pct:.1f}%) | "
            f"{row.performance_1yr:.2f}% | {row.volatility:.2f}% | "
            f"{row.expense_ratio:.2f}% | {row.dividend_yield:.2f}% | "
            f"{row.description} |"
        )
    etf_summary = "\n".join(etf_rows)
    # Portfolio metrics
    avg_expense = (picks["expense_ratio"] * picks["allocation"]).sum() / total_allocation
    avg_volatility = (picks["volatility"] * picks["allocation"]).sum() / total_allocation
    avg_performance = (picks["performance_1yr"] * picks["allocation"]).sum() / total_allocation
    sector_overlap = etf_data.get_sector_overlap(picks['ticker'].tolist())
    portfolio_metrics = (
        f"\nPortfolio Metrics:\n"
        f"• Average 1-Year Return: {avg_performance:.2f}%\n"
        f"• Average Expense Ratio: {avg_expense:.2f}%\n"
        f"• Portfolio Volatility: {avg_volatility:.2f}%\n"
        f"• Sector Concentration: {sector_overlap:.1%}\n"
    )
    # Add few-shot example
    example = """
Example Response Format:

| Ticker | Name | Allocation | 1Y Return | Volatility | Expense Ratio | Dividend Yield | Description |
|--------|-------------------------------|------------|-----------|------------|---------------|---------------|-------------|
| BOTZ   | Global X Robotics & AI ETF | $50,000 (50.0%) |  15.20% |    18.00% | 0.68% | 0.00% | Provides exposure to companies involved in robotics, AI, and automation across developed markets |
| ARKK   | ARK Innovation ETF | $50,000 (50.0%) |  12.40% |    25.00% | 0.75% | 0.00% | Invests in companies that benefit from disruptive innovation across multiple sectors |

Portfolio Metrics:
• Average 1-Year Return: 13.80%
• Average Expense Ratio: 0.72%
• Portfolio Volatility: 21.50%
• Sector Concentration: 45.0%

This portfolio offers strong exposure to AI and innovation with a balanced allocation between robotics (BOTZ) and disruptive tech (ARKK). The 21.5% volatility is appropriate for medium risk tolerance. Consider adding a small allocation to a low-volatility tech ETF to reduce overall risk.
"""
    # Check for negative performance
    all_negative = all(picks["performance_1yr"] < 0)
    if all_negative:
        warning = (
            "⚠️ Note: All selected ETFs have had negative 1-year returns. "
            "This may reflect short-term macroeconomic headwinds. "
            "Consider diversifying further if concerned about short-term performance.\n\n"
        )
    else:
        warning = ""
    return (
        f"{warning}"
        f"Based on your ${user_prefs['budget']:,} investment in {', '.join(user_prefs['themes'])} "
        f"with {user_prefs['risk_level']} risk tolerance, here's your recommended portfolio as a Markdown table:\n\n"
        f"{etf_summary}\n{portfolio_metrics}\n"
        f"Provide a concise explanation of this portfolio following this exact format:\n"
        f"{example}\n"
        f"Keep the explanation under 300 words and maintain the Markdown table format exactly as shown."
    )

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
                " I couldn't find ETFs matching those themes. "
                "Try different keywords or a different risk level. "
                "For example, you could try:\n"
                "• Different theme keywords\n"
                "• A broader risk level\n"
                "• A different investment amount"
            )
        
        # ---- Portfolio construction ----
        portfolio = allocate_portfolio(budget, candidate_etfs)
        
        # ---- LLM explanation ----
        prompt = build_explanation_prompt(
            {"themes": themes, "budget": budget, "risk_level": risk_level},
            portfolio
        )
        
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
    with gr.Blocks(title="AI Portfolio Advisor") as demo:
        gr.Markdown("""
        # AI Portfolio Advisor
        
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
            #retry_btn=None,
            #undo_btn=None,
            #clear_btn="Clear Chat",
        )
        
        gr.Markdown("""
        ---
        *Powered by OpenAI + LangChain + Gradio*
        
        This is a demonstration project and not financial advice. 
        Always consult with a financial advisor before making investment decisions.
        """)
    
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
