"""Fetch dated source observations; never substitute model defaults for market data."""
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SERIES = [
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

def fetch(spec):
    sid, label, unit, max_age = spec
    result = dict(id=sid, label=label, unit=unit, max_age_days=max_age,
                  source='https://fred.stlouisfed.org/series/' + sid)
    try:
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=' + sid
        with urlopen(Request(url, headers={'User-Agent': 'CollapseMarketMonitor/2.0'}), timeout=45) as r:
            rows = list(csv.DictReader(io.StringIO(r.read().decode())))
        valid = []
        today = datetime.now(timezone.utc).date()
        for row in rows:
            date = row.get('observation_date', row.get('DATE'))
            try:
                value = float(row[sid])
                obs = datetime.strptime(date, '%Y-%m-%d').date()
                if obs <= today:
                    valid.append((date, value))
            except (ValueError, TypeError):
                continue
        date, value = sorted(valid)[-1]
        result.update(value=value, observation_date=date, status='available')
    except Exception as exc:
        result.update(value=None, observation_date=None, status='unavailable', error=type(exc).__name__)
    return result

def main():
    with ThreadPoolExecutor(max_workers=6) as pool:
        observations = list(pool.map(fetch, SERIES))
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
