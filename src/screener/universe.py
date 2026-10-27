"""
src/screener/universe.py
Universe definition containing 40 liquid US mega-cap stock tickers.
"""

from typing import List, Dict, Any
import pandas as pd

# 40 liquid US mega-cap tickers across key market sectors
MEGA_CAP_UNIVERSE: List[str] = [
    # Technology / Communication
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "CSCO",
    # Financial Services
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "C", "BLK",
    # Healthcare / Pharma
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "DHR", "AMGN", "ABT",
    # Consumer Staples & Discretionary
    "WMT", "PG", "COST", "HD", "KO", "PEP", "MCD", "DIS", "NKE", "NFLX"
]

# Comprehensive company metadata mapping ticker -> {name, sector, industry}
MEGA_CAP_COMPANIES: Dict[str, Dict[str, str]] = {
    # Technology / Communication
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics & Hardware",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Enterprise Software & Cloud Systems",
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Search Engine & Digital Advertising",
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "sector": "Consumer Discretionary",
        "industry": "E-Commerce & Cloud Infrastructure (AWS)",
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "sector": "Technology",
        "industry": "Accelerated Computing & AI Hardware",
    },
    "META": {
        "name": "Meta Platforms Inc.",
        "sector": "Communication Services",
        "industry": "Social Media Platforms & Metaverse",
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "sector": "Consumer Discretionary",
        "industry": "Electric Vehicles & Clean Energy",
    },
    "AVGO": {
        "name": "Broadcom Inc.",
        "sector": "Technology",
        "industry": "Semiconductors & Infrastructure Software",
    },
    "ORCL": {
        "name": "Oracle Corporation",
        "sector": "Technology",
        "industry": "Enterprise Database & Cloud Infrastructure",
    },
    "CSCO": {
        "name": "Cisco Systems Inc.",
        "sector": "Technology",
        "industry": "Networking Hardware & Cybersecurity",
    },
    # Financial Services
    "BRK-B": {
        "name": "Berkshire Hathaway Inc.",
        "sector": "Financial Services",
        "industry": "Multi-Sector Holding & Insurance",
    },
    "JPM": {
        "name": "JPMorgan Chase & Co.",
        "sector": "Financial Services",
        "industry": "Global Investment & Commercial Banking",
    },
    "V": {
        "name": "Visa Inc.",
        "sector": "Financial Services",
        "industry": "Digital Payment Processing Network",
    },
    "MA": {
        "name": "Mastercard Incorporated",
        "sector": "Financial Services",
        "industry": "Global Payments Technology & Network",
    },
    "BAC": {
        "name": "Bank of America Corp.",
        "sector": "Financial Services",
        "industry": "Consumer & Commercial Banking",
    },
    "WFC": {
        "name": "Wells Fargo & Company",
        "sector": "Financial Services",
        "industry": "Retail & Commercial Banking",
    },
    "MS": {
        "name": "Morgan Stanley",
        "sector": "Financial Services",
        "industry": "Wealth Management & Investment Banking",
    },
    "GS": {
        "name": "The Goldman Sachs Group Inc.",
        "sector": "Financial Services",
        "industry": "Investment Banking & Capital Markets",
    },
    "C": {
        "name": "Citigroup Inc.",
        "sector": "Financial Services",
        "industry": "Global Institutional & Consumer Banking",
    },
    "BLK": {
        "name": "BlackRock Inc.",
        "sector": "Financial Services",
        "industry": "Asset Management & Investment Systems",
    },
    # Healthcare / Pharma
    "UNH": {
        "name": "UnitedHealth Group Inc.",
        "sector": "Healthcare",
        "industry": "Managed Healthcare & Health Insurance",
    },
    "JNJ": {
        "name": "Johnson & Johnson",
        "sector": "Healthcare",
        "industry": "Pharmaceuticals & Medical Technology",
    },
    "LLY": {
        "name": "Eli Lilly and Company",
        "sector": "Healthcare",
        "industry": "Pharmaceuticals & Metabolic Therapeutics",
    },
    "ABBV": {
        "name": "AbbVie Inc.",
        "sector": "Healthcare",
        "industry": "Biopharmaceuticals & Immunology",
    },
    "MRK": {
        "name": "Merck & Co. Inc.",
        "sector": "Healthcare",
        "industry": "Pharmaceuticals & Oncology Therapeutics",
    },
    "PFE": {
        "name": "Pfizer Inc.",
        "sector": "Healthcare",
        "industry": "Biopharmaceuticals & Vaccines",
    },
    "TMO": {
        "name": "Thermo Fisher Scientific Inc.",
        "sector": "Healthcare",
        "industry": "Life Sciences Tools & Diagnostic Instruments",
    },
    "DHR": {
        "name": "Danaher Corporation",
        "sector": "Healthcare",
        "industry": "Medical Diagnostics & Life Sciences",
    },
    "AMGN": {
        "name": "Amgen Inc.",
        "sector": "Healthcare",
        "industry": "Biotechnology Therapeutics",
    },
    "ABT": {
        "name": "Abbott Laboratories",
        "sector": "Healthcare",
        "industry": "Medical Devices, Diagnostics & Nutrition",
    },
    # Consumer Staples & Discretionary
    "WMT": {
        "name": "Walmart Inc.",
        "sector": "Consumer Staples",
        "industry": "Retail Hypermarkets & Omnichannel Grocery",
    },
    "PG": {
        "name": "The Procter & Gamble Company",
        "sector": "Consumer Staples",
        "industry": "Consumer Household & Personal Products",
    },
    "COST": {
        "name": "Costco Wholesale Corporation",
        "sector": "Consumer Staples",
        "industry": "Warehouse Membership Wholesale Retail",
    },
    "HD": {
        "name": "The Home Depot Inc.",
        "sector": "Consumer Discretionary",
        "industry": "Home Improvement Retail",
    },
    "KO": {
        "name": "The Coca-Cola Company",
        "sector": "Consumer Staples",
        "industry": "Non-Alcoholic Beverages",
    },
    "PEP": {
        "name": "PepsiCo Inc.",
        "sector": "Consumer Staples",
        "industry": "Beverages & Packaged Snack Foods",
    },
    "MCD": {
        "name": "McDonald's Corporation",
        "sector": "Consumer Discretionary",
        "industry": "Fast-Food Restaurant Franchising",
    },
    "DIS": {
        "name": "The Walt Disney Company",
        "sector": "Communication Services",
        "industry": "Entertainment, Theme Parks & Streaming",
    },
    "NKE": {
        "name": "NIKE Inc.",
        "sector": "Consumer Discretionary",
        "industry": "Athletic Footwear, Apparel & Equipment",
    },
    "NFLX": {
        "name": "Netflix Inc.",
        "sector": "Communication Services",
        "industry": "Subscription Video Streaming Entertainment",
    },
}


def get_universe() -> List[str]:
    """Returns a copy of the mega-cap stock ticker universe."""
    return list(MEGA_CAP_UNIVERSE)


def get_company_info(ticker: str) -> Dict[str, str]:
    """
    Returns company name, sector, and industry for a ticker symbol.
    Falls back to ticker symbol if unlisted.
    """
    return MEGA_CAP_COMPANIES.get(
        ticker.upper(),
        {
            "name": ticker.upper(),
            "sector": "General Equities",
            "industry": "Diversified Market Equities",
        },
    )


def get_company_name(ticker: str) -> str:
    """Returns the full corporate name for a ticker symbol."""
    return get_company_info(ticker)["name"]


def get_company_sector(ticker: str) -> str:
    """Returns the primary industry sector for a ticker symbol."""
    return get_company_info(ticker)["sector"]


def get_company_industry(ticker: str) -> str:
    """Returns the sub-industry classification for a ticker symbol."""
    return get_company_info(ticker)["industry"]


def get_universe_dataframe() -> pd.DataFrame:
    """
    Returns a pandas DataFrame representation of the full universe directory
    containing Ticker, Company Name, Sector, and Industry Focus.
    """
    records = []
    for ticker in MEGA_CAP_UNIVERSE:
        info = get_company_info(ticker)
        records.append({
            "Ticker": ticker,
            "Company Name": info["name"],
            "Sector": info["sector"],
            "Industry Focus": info["industry"],
        })
    return pd.DataFrame(records)
