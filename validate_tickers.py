"""
Validate which tickers return data from yfinance.
Reads data/all_tickers_tesing.csv, writes data/valid_tickers.csv.
Run: python validate_tickers.py
"""
import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from get_price_data import fetch_latest_data

_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(_HERE, 'data', 'all_tickers_tesing.csv')
OUTPUT_CSV = os.path.join(_HERE, 'data', 'valid_tickers.csv')
WORKERS = 20
TEST_PERIOD = '60d'   # short enough to be fast, long enough to confirm the ticker is live


def check_ticker(ticker: str) -> tuple[str, bool]:
    df = fetch_latest_data(ticker, period=TEST_PERIOD)
    return ticker, df is not None and len(df) >= 10


def main():
    tickers = pd.read_csv(INPUT_CSV, header=None).iloc[:, 0].dropna().str.strip().tolist()
    total = len(tickers)
    print(f"Checking {total} tickers with {WORKERS} workers (period={TEST_PERIOD})...")

    valid = []
    failed = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(check_ticker, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, ok = fut.result()
            (valid if ok else failed).append(ticker)
            if i % 100 == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                print(f"  {i}/{total} checked | valid={len(valid)} | "
                      f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining",
                      flush=True)

    valid.sort()
    pd.Series(valid).to_csv(OUTPUT_CSV, index=False, header=False)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Valid : {len(valid)}")
    print(f"Failed: {len(failed)}")
    print(f"Saved : {OUTPUT_CSV}")

    # breakdown by exchange suffix
    for suffix in ['.hk', '.SI', '.sz', '.SS']:
        n = sum(1 for t in valid if t.lower().endswith(suffix.lower()))
        if n:
            print(f"  {suffix}: {n}")


if __name__ == '__main__':
    main()
