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
    out = df[["Close"]].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    return out.dropna().sort_index()


def _start(days: int) -> str:
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()


def history_kr(code: str, days: int = 420) -> pd.DataFrame:
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


def history_us(ticker: str, days: int = 420) -> pd.DataFrame:
    try:
        import yfinance as yf
        return _clean(yf.download(ticker.strip().upper(), start=_start(days), progress=False, auto_adjust=True))
    except Exception:
        return pd.DataFrame(columns=["Close"])


def history_fx(days: int = 420) -> pd.DataFrame:
    """원/달러 환율."""
    try:
        import yfinance as yf
        return _clean(yf.download("KRW=X", start=_start(days), progress=False, auto_adjust=True))
    except Exception:
        return pd.DataFrame(columns=["Close"])


def kakao_value(px_usd: pd.DataFrame, fx: pd.DataFrame, per_day_krw: float, start: str):
    """매 거래일 per_day_krw 원어치를 소수점으로 샀다고 보고 원금과 평가액을 계산한다.
    반환: (원금, 평가액, 회차) 또는 계산 불가 시 None."""
    if px_usd.empty or fx.empty:
        return None
    s = pd.to_datetime(start, errors="coerce")
    if pd.isna(s):
        return None
    px = px_usd["Close"][px_usd.index >= s]
    if px.empty:
        return None
    rate = fx["Close"].reindex(px.index, method="ffill").bfill()
    shares = (per_day_krw / (px * rate)).sum()
    n = len(px)
    value = shares * px.iloc[-1] * rate.iloc[-1]
    return float(per_day_krw * n), float(value), int(n)
