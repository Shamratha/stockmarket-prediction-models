"""Stock Market local dashboard.

Serves the results produced by scripts/run_*.py (tables + charts in output/)
plus a live quick-forecast endpoint. Local use only:

    .venv\\Scripts\\python -m uvicorn app:app --port 8600
"""

import glob
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'output')

app = FastAPI(title='Stock Market', docs_url='/api/docs')


@app.get('/api/summary')
def summary():
    tickers = sorted(
        os.path.basename(f).replace('forecasting_results_', '').replace('.csv', '')
        for f in glob.glob(os.path.join(OUT, 'forecasting_results_*.csv'))
    )
    forecasting, agents = {}, {}
    for tk in tickers:
        fdf = pd.read_csv(os.path.join(OUT, f'forecasting_results_{tk}.csv'))
        forecasting[tk] = fdf.replace({np.nan: None}).to_dict(orient='records')
        apath = os.path.join(OUT, f'agent_results_{tk}.csv')
        if os.path.exists(apath):
            agents[tk] = pd.read_csv(apath).to_dict(orient='records')

    charts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, '*.png')))
    return {
        'tickers': tickers,
        'forecasting': forecasting,
        'agents': agents,
        'charts': charts,
    }


@app.get('/api/quick-forecast')
def quick_forecast(ticker: str = 'GOOG', days: int = 30):
    """Fast ARIMA forecast for the interactive widget (~2s)."""
    if days < 1 or days > 90:
        raise HTTPException(400, 'days must be 1-90')
    from stockmarket.data import load_prices, log_returns
    from stockmarket.forecasting.baselines import ARIMAForecaster

    try:
        df = load_prices(ticker.upper())
    except Exception as e:
        raise HTTPException(404, f'no data for {ticker}: {e}')

    close = df['Close']
    rets = log_returns(close).values.astype('float32')
    model = ARIMAForecaster()
    model.fit(rets)
    fc = model.forecast_recursive(rets, days)
    prices = close.values[-1] * np.exp(np.cumsum(fc))

    hist_n = 90
    return {
        'ticker': ticker.upper(),
        'last_close': float(close.values[-1]),
        'history': {
            'dates': [d.strftime('%Y-%m-%d') for d in close.index[-hist_n:]],
            'prices': [float(p) for p in close.values[-hist_n:]],
        },
        'forecast': [float(p) for p in prices],
        'note': 'ARIMA recursive scenario — uncertainty compounds each step; not a prediction.',
    }


app.mount('/output', StaticFiles(directory=OUT), name='output')


@app.get('/')
def index():
    return FileResponse(os.path.join(ROOT, 'static', 'index.html'))
