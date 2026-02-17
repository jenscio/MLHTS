import yfinance as yf
import pandas as pd

UNIVERSE = {
    # Commodities
    'GLD':  'Gold',
    'SLV':  'Silver',
    'PPLT': 'Platinum',
    'USO':  'Crude Oil WTI',
    'BNO':  'Brent Crude',
    'UNG':  'Natural Gas',
    'DBC':  'Broad Commodity',
    'CPER': 'Copper',
    'CORN': 'Corn',
    'WEAT': 'Wheat',
    # Equities
    'SPY':  'S&P 500',
    'QQQ':  'Nasdaq 100',
    'IWM':  'Russell 2000',
    'EFA':  'MSCI EAFE',
    'EEM':  'MSCI Emerging',
    'EWJ':  'MSCI Japan',
    'EWL':  'MSCI Switzerland',
    'EWZ':  'MSCI Brazil',
    'EWG':  'MSCI Germany',
    'EWU':  'MSCI UK',
    'FXI':  'China Large Cap',
    'CSSMI.SW': 'Swiss SMI (CHF)',
    # Fixed Income
    'SHY':  '1-3Y Treasury',
    'IEF':  '7-10Y Treasury',
    'TLT':  '20+Y Treasury',
    'LQD':  'IG Corporate',
    'HYG':  'HY Corporate',
    'BNDX': 'Intl Bonds',
    'TIP':  'TIPS',
    'EMB':  'EM Bonds',
    # Currencies
    'UUP':  'US Dollar Index',
    'FXE':  'Euro',
    'FXB':  'British Pound',
    'FXY':  'Japanese Yen',
    'FXA':  'Australian Dollar',
    'FXF':  'Swiss Franc',
    'FXC':  'Canadian Dollar',
    'CEW':  'EM Currency Basket',
}

ASSET_CLASSES = {
    'GLD': 'Commodity', 'SLV': 'Commodity', 'PPLT': 'Commodity',
    'USO': 'Commodity', 'BNO': 'Commodity', 'UNG': 'Commodity',
    'DBC': 'Commodity', 'CPER': 'Commodity', 'CORN': 'Commodity',
    'WEAT': 'Commodity',
    'SPY': 'Equity', 'QQQ': 'Equity', 'IWM': 'Equity',
    'EFA': 'Equity', 'EEM': 'Equity', 'EWJ': 'Equity',
    'EWL': 'Equity', 'EWZ': 'Equity', 'EWG': 'Equity',
    'EWU': 'Equity', 'FXI': 'Equity', 'CSSMI.SW': 'Equity',
    'SHY': 'Fixed Income', 'IEF': 'Fixed Income',
    'TLT': 'Fixed Income', 'LQD': 'Fixed Income',
    'HYG': 'Fixed Income', 'BNDX': 'Fixed Income',
    'TIP': 'Fixed Income', 'EMB': 'Fixed Income',
    'UUP': 'Currency', 'FXE': 'Currency', 'FXB': 'Currency',
    'FXY': 'Currency', 'FXA': 'Currency', 'FXF': 'Currency',
    'FXC': 'Currency', 'CEW': 'Currency',
}

def download_universe(start='2006-01-01', end='2024-12-31'):
    """Download and clean price data for full universe"""
    tickers = list(UNIVERSE.keys())
    data = yf.download(tickers, start=start, end=end)
    prices = data['Close']

    # Report coverage
    print(f"\n{'Ticker':<12} {'Name':<28} {'Class':<14} {'Start':<12} {'End':<12} {'Days':>6}")
    print('-' * 90)
    for ticker in UNIVERSE:
        try:
            valid = prices[ticker].dropna()
            if len(valid) > 0:
                print(f"{ticker:<12} {UNIVERSE[ticker]:<28} {ASSET_CLASSES[ticker]:<14} "
                      f"{valid.index[0].date()!s:<12} {valid.index[-1].date()!s:<12} {len(valid):>6}")
            else:
                print(f"{ticker:<12} {UNIVERSE[ticker]:<28} NO DATA")
        except KeyError:
            print(f"{ticker:<12} {UNIVERSE[ticker]:<28} DOWNLOAD FAILED")

    return prices

if __name__ == '__main__':
    prices = download_universe()
    prices.to_csv('data/raw/prices.csv')
    print(f"\nSaved {len(UNIVERSE)} assets to data/raw/prices.csv")