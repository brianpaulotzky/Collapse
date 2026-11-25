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
    ('GDP', 'Nominal GDP', 'USD billions', 240),
    ('BOGZ1LM883164115Q', 'Domestic public corporate equity market value', 'USD millions', 240),
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

def valuation_inputs():
    results = []
    def item(sid, label, source, date, value, age):
        return dict(id=sid, label=label, source=source, observation_date=date,
                    value=value, max_age_days=age, unit='ratio' if sid=='CAPE' else 'USD millions' if sid=='MARGIN_DEBT' else '%', status='available')
    try:
        eq = {r.get('observation_date', r.get('DATE')): float(r['BOGZ1LM883164115Q']) for r in retrieve('BOGZ1LM883164115Q') if r.get('BOGZ1LM883164115Q') not in (None,'.','')}
        gdp = {r.get('observation_date', r.get('DATE')): float(r['GDP']) for r in retrieve('GDP') if r.get('GDP') not in (None,'.','')}
        d = max(k for k in eq.keys() & gdp.keys() if gdp[k]>0 and eq[k]>0 and k<=str(datetime.now(timezone.utc).date()))
        results.append(item('BUFFETT','Public domestic equity / GDP (Fed proxy)','https://fred.stlouisfed.org/series/BOGZ1LM883164115Q',d,eq[d]/gdp[d]/10,240))
    except Exception as exc:
        print('BUFFETT:',type(exc).__name__,flush=True)
    try:
        # Shiller workbook: use actual published CAPE, never padded zero values.
        import pandas as pd
        url='https://www.econ.yale.edu/~shiller/data/ie_data.xls'
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as r:
            frame=pd.read_excel(io.BytesIO(r.read()),sheet_name='Data',header=None,engine='xlrd')
        found=[]
        for row in frame.itertuples(index=False,name=None):
            try:
                raw=float(row[0]); year=int(raw); month=round((raw-year)*100)
                date=datetime(year,month,1).date()
                value=float(row[12])
                if 0<value<150 and date<=datetime.now(timezone.utc).date(): found.append((str(date),value))
            except (ValueError,TypeError,OverflowError,IndexError): pass
        date,value=max(found)
        results.append(item('CAPE','Shiller CAPE',url,date,value,100))
    except Exception as exc:
        print('CAPE:',type(exc).__name__,flush=True)
    try:
        import pandas as pd
        from urllib.parse import urljoin
        import re
        page='https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics'
        with urlopen(Request(page,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as r:
            markup=r.read().decode('utf-8')
        links=re.findall(r'href=["\x27]([^"\x27]+\\.xlsx(?:\\?[^"\x27]*)?)["\x27]',markup,re.I)
        url=urljoin(page,next(x for x in links if 'margin' in x.lower()))
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as r:
            sheet=pd.read_excel(io.BytesIO(r.read()),header=None)
        debt={}
        for row in sheet.itertuples(index=False,name=None):
            try:
                if isinstance(row[0],(int,float)): continue
                d=pd.to_datetime(row[0],errors='raise').strftime('%Y-%m')
                v=float(str(row[1]).replace(',',''))
                if 0<v and d<=datetime.now(timezone.utc).strftime('%Y-%m'): debt[d]=v
            except (ValueError,TypeError,IndexError): pass
        series=pd.Series(debt,dtype=float).sort_index()
        series.index=pd.PeriodIndex(series.index,freq='M')
        series=series.asfreq('M')
        if len(series.dropna())<36 or pd.isna(series.iloc[-1]): raise ValueError('Insufficient margin history')
        date=str(series.index[-1].start_time.date())
        results.append(item('MARGIN_DEBT','FINRA margin debit balances (USD millions)',url,date,float(series.iloc[-1]),100))
        growth=series.pct_change(12,fill_method=None)
        contraction=(-series.pct_change(3,fill_method=None)).clip(lower=0)*growth.shift(3).clip(lower=0)
        def percentile(x):
            x=x.dropna()
            if len(x)<24: raise ValueError('Insufficient history')
            return float(100*(x.iloc[:-1]<=x.iloc[-1]).mean()) if x.iloc[-1]>0 else 0.0
        if pd.notna(contraction.iloc[-1]):
            results.append(item('MARGIN_CONTRACTION_PCTL','Margin contraction after expansion percentile',url,date,percentile(contraction),100))
        # Normalize against matched-quarter public equity market value, no future quarters.
        eq_series=pd.Series(eq,dtype=float)
        eq_series.index=pd.PeriodIndex(pd.to_datetime(eq_series.index),freq='Q')
        ratios=[]
        for period,amount in series.dropna().items():
            q=period.asfreq('Q')-1
            if q in eq_series.index and eq_series[q]>0: ratios.append((period,amount/eq_series[q]))
        if len(ratios)>=25 and ratios[-1][0]==series.index[-1]:
            values=pd.Series([v for _,v in ratios])
            results.append(item('MARGIN_INTENSITY_PCTL','Margin debt / lagged public equity percentile',url,date,percentile(values),100))
    except Exception as exc:
        print('FINRA:',type(exc).__name__,flush=True)
    return results

def main():
    with ThreadPoolExecutor(max_workers=6) as pool:
        observations = list(pool.map(fetch, SERIES))
    observations.extend(valuation_inputs())
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
