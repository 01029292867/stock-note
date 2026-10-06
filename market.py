"""시세 가져오기. 무료 데이터라 지연되거나 실패할 수 있어서 항상 빈 결과를 안전하게 돌려준다."""
import datetime as dt
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Close"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return pd.DataFrame(columns=["Close"])
    cols = ["Close"] + (["Volume"] if "Volume" in df.columns else [])
    out = df[cols].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["Close"]).sort_index()


def _start(days: int) -> str:
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


def history_kr(code: str, days: int = 800) -> pd.DataFrame:
    """국내 종목 일봉 종가. FinanceDataReader 먼저, 실패하면 야후(.KS/.KQ)로 시도."""
    code = str(code).strip().zfill(6)
    try:
        import FinanceDataReader as fdr
        df = _clean(fdr.DataReader(code, _start(days)))
        if len(df) > 0:
            return df
    except Exception:
        pass
    try:
        import yfinance as yf
        for suffix in (".KS", ".KQ"):
            df = _clean(yf.download(code + suffix, start=_start(days), progress=False, auto_adjust=True))
            if len(df) > 0:
                return df
    except Exception:
        pass
    return pd.DataFrame(columns=["Close"])


def history_us(ticker: str, days: int = 1000) -> pd.DataFrame:
    try:
        import yfinance as yf
        return _clean(yf.download(ticker.strip().upper(), start=_start(days), progress=False, auto_adjust=True))
    except Exception:
        return pd.DataFrame(columns=["Close"])


def history_fx(days: int = 1000) -> pd.DataFrame:
    """원/달러 환율."""
    try:
        import yfinance as yf
        return _clean(yf.download("KRW=X", start=_start(days), progress=False, auto_adjust=True))
    except Exception:
        return pd.DataFrame(columns=["Close"])


def kakao_value(px_usd: pd.DataFrame, fx: pd.DataFrame, per_day_krw: float, start=None, count=None, asof=None):
    """매 미국 거래일 per_day_krw 원어치를 소수점으로 샀다고 보고 원금과 평가액을 계산한다.

    횟수(count)와 기준일(asof)로 시작 시점을 정한다: 기준일까지 count번 샀다면 그 count번째 전 거래일이 시작일이다.
    기준일 이후에는 거래일마다 횟수가 자동으로 하나씩 늘어난다. count가 없으면 시작일(start)을 쓴다.
    반환: {'cost','value','n','start','short'} 또는 계산할 수 없으면 None. short=True는 시세 기록이 모자라 일부 회차가 빠졌다는 뜻."""
    if px_usd.empty or fx.empty:
        return None
    px = px_usd["Close"]
    rate_all = fx["Close"].reindex(px.index, method="ffill").bfill()
    short = False
    if count is not None and not pd.isna(count):
        a = pd.to_datetime(asof, errors="coerce")
        a = pd.Timestamp.today().normalize() if pd.isna(a) else a
        pos = int(px.index.searchsorted(a, side="right")) - 1          # 기준일 이전(포함) 마지막 거래일
        c = int(count)
        i0 = pos - (c - 1) if c >= 1 else pos + 1
        if i0 < 0:
            short, i0 = True, 0
    else:
        s = pd.to_datetime(start, errors="coerce")
        if pd.isna(s):
            return None
        i0 = int(px.index.searchsorted(s, side="left"))
    sel = px.iloc[i0:]
    if sel.empty:
        return {"cost": 0.0, "value": 0.0, "n": 0, "start": None, "short": short}
    rate = rate_all.iloc[i0:]
    shares = (per_day_krw / (sel * rate)).sum()
    n = len(sel)
    return {"cost": float(per_day_krw * n), "value": float(shares * sel.iloc[-1] * rate.iloc[-1]), "n": int(n),
            "start": sel.index[0].date().isoformat(), "short": short}


def avg_trading_value_eok(df: pd.DataFrame, n: int = 20):
    """최근 n거래일 평균 거래대금(억원). 거래량이 없으면 None."""
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    v = (df["Close"] * df["Volume"]).tail(n).dropna()
    return float(v.mean() / 1e8) if len(v) else None


def history_index(days: int = 1000) -> pd.DataFrame:
    """코스피 지수 종가(판단 결과를 시장과 비교할 때 쓴다)."""
    try:
        import FinanceDataReader as fdr
        df = _clean(fdr.DataReader("KS11", _start(days)))
        if len(df) > 0:
            return df
    except Exception:
        pass
    try:
        import yfinance as yf
        return _clean(yf.download("^KS11", start=_start(days), progress=False, auto_adjust=True))
    except Exception:
        return pd.DataFrame(columns=["Close"])
