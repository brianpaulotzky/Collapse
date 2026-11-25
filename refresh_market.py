"""Fetch dated source observations; never substitute model defaults for market data."""
import csv
import io
import json
import math
import os
import time
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SERIES = [
    ('T10Y3M', '10Y–3M Treasury spread', 'percentage points', 7),
    ('BAMLH0A0HYM2', 'High-yield credit spread', 'percentage points', 7),
    ('VXVCLS', 'CBOE 3-month volatility index', 'index points', 7),
    ('DFII10', '10-year real Treasury yield', '%', 7),
    ('GDP', 'Nominal GDP', 'USD billions', 150),
    ('NCBEILQ027S', 'Nonfinancial corporate equity liabilities (proxy only)', 'USD millions', 150),
    ('DJIA', 'Dow Jones Industrial Average', 'index points', 7),
    ('NASDAQ100', 'Nasdaq 100', 'index points', 7),
    ('WILL5000INDFC', 'Wilshire 5000', 'index points', 7),
    ('NASDAQNMS', 'Nasdaq Global Market Composite', 'index points', 7),
    ('SP500', 'S&P 500', 'index points', 7),
    ('NASDAQCOM', 'Nasdaq Composite', 'index points', 7),
    ('VIXCLS', 'VIX', 'index points', 7),
    ('DGS10', '10-year Treasury yield', '%', 7),
    ('T10Y2Y', '10Y–2Y Treasury spread', 'percentage points', 7),
    ('BAA10YM', 'Baa–10Y credit spread', 'percentage points', 65),
    ('DCOILWTICO', 'WTI crude oil', 'USD/barrel', 10),
    ('NFCI', 'Chicago Fed financial conditions', 'index', 21),
    ('STLFSI4', 'St. Louis financial stress', 'index', 21),
    ('UMCSENT', 'Michigan consumer sentiment', 'index', 65),
    ('USEPUINDXM', 'US policy uncertainty', 'index', 65),
]

def retrieve(sid):
    key = os.getenv('FRED_API_KEY')
    if key:
        try:
            url = 'https://api.stlouisfed.org/fred/series/observations?' + urlencode(dict(series_id=sid, api_key=key, file_type='json', sort_order='desc', limit=100))
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=20) as r:
                return [{'observation_date': x['date'], sid: x['value']} for x in json.load(r)['observations']]
        except Exception as exc:
            print(f'{sid}: official API failed ({type(exc).__name__})', flush=True)
    for attempt in range(3):
        try:
            url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?' + urlencode(dict(id=sid, cosd=str(datetime.now(timezone.utc).date().replace(year=datetime.now(timezone.utc).year-1, day=1))))
            with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'text/csv,*/*'}), timeout=20) as r:
                return list(csv.DictReader(io.StringIO(r.read().decode('utf-8-sig'))))
        except Exception as exc:
            print(f'{sid}: CSV attempt {attempt+1}: {type(exc).__name__}: {exc}', flush=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def fetch(spec):
    sid, label, unit, max_age = spec
    result = dict(id=sid, label=label, unit=unit, max_age_days=max_age,
                  source='https://fred.stlouisfed.org/series/' + sid)
    try:
        rows = retrieve(sid)
        valid = []
        today = datetime.now(timezone.utc).date()
        for row in rows:
            date = row.get('observation_date', row.get('DATE'))
            try:
                value = float(row[sid])
                obs = datetime.strptime(date, '%Y-%m-%d').date()
                if obs <= today and math.isfinite(value):
                    valid.append((date, value))
            except (ValueError, TypeError, KeyError):
                continue
        valid = sorted(dict(valid).items())
        date, value = valid[-1]
        result.update(value=value, observation_date=date, status='available')
        if len(valid) >= 2 and valid[-2][1] != 0:
            previous_date, previous_value = valid[-2]
            result.update(previous_date=previous_date, previous_value=previous_value,
                          change_pct=100 * (value / previous_value - 1))
    except Exception as exc:
        print(f'{sid}: {type(exc).__name__}', flush=True)
        result.update(value=None, observation_date=None, status='unavailable', error=type(exc).__name__)
    return result

def main():
    with ThreadPoolExecutor(max_workers=6) as pool:
        observations = list(pool.map(fetch, SERIES))
    dest = ROOT / 'market_snapshot.json'
    if not any(x['status'] == 'available' for x in observations):
        raise SystemExit('No source observations could be retrieved; existing snapshot preserved')
    try:
        previous = {x['id']: x for x in json.loads(dest.read_text())['observations']}
    except (OSError, ValueError, KeyError):
        previous = {}
    for i, item in enumerate(observations):
        old = previous.get(item['id'], {})
        if item['status'] != 'available' and old.get('value') is not None:
            observations[i] = dict(old, status='cached', error=item.get('error'))
    snapshot = dict(fetched_at=datetime.now(timezone.utc).isoformat(), observations=observations,
                    note='Latest published observations, not real-time quotes. Monthly dates identify the observation period; releases can lag and be revised.')
    dest = ROOT / 'market_snapshot.json'
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
    temp.replace(dest)
    if not any(x['status'] == 'available' for x in observations):
        raise SystemExit('No source observations could be retrieved')
    print('\n'.join(f"{x['id']}: {x['value']} ({x['observation_date']})" for x in observations))

if __name__ == '__main__':
    main()
