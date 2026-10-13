"""코인 평가(보기 전용): 거래소, 코인 약어, 수량, 평균단가를 직접 적으면 원화 시세로 평가금액을 계산한다.
시세는 업비트 공개 API(없는 코인은 빗썸)에서 가져오며 API 키는 필요 없다. 주문은 하지 않는다."""
VERSION = 1

import numpy as np
import pandas as pd
import requests

COLS = ["거래소", "코인", "수량", "평균단가", "메모"]
EXCHANGES = ["업비트", "빗썸", "코인원", "바이낸스", "기타"]
UPBIT_ALL = "https://api.upbit.com/v1/market/all"
UPBIT_TICKER = "https://api.upbit.com/v1/ticker"
BITHUMB = "https://api.bithumb.com/public/ticker/{sym}_KRW"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "application/json"}
_BLANK = {"", "NONE", "NAN", "NAT"}


def norm_symbol(x):
    s = str(x if x is not None else "").strip().upper()
    for p in ("KRW-", "KRW "):
        if s.startswith(p):
            s = s[len(p):]
    for q in ("/KRW", "-KRW", "_KRW"):
        if s.endswith(q):
            s = s[: -len(q)]
    return "" if s in _BLANK else s


def clean(df):
    """입력 표를 정리한다: 코인 약어를 대문자로, 수량·평균단가를 숫자로. 코인 약어가 없는 줄은 뺀다."""
    d = df.copy() if df is not None else pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in d.columns:
            d[c] = None
    d = d[COLS].copy()
    d["코인"] = d["코인"].map(norm_symbol)
    d = d[d["코인"] != ""]
    d["수량"] = pd.to_numeric(d["수량"], errors="coerce").fillna(0.0)
    d["평균단가"] = pd.to_numeric(d["평균단가"], errors="coerce")
    d["거래소"] = d["거래소"].fillna("").astype(str).replace({"None": "", "nan": ""}).where(lambda s: s != "", "기타")
    d["메모"] = d["메모"].fillna("").astype(str).replace({"None": "", "nan": ""})
    return d.reset_index(drop=True)


def blank_symbol_rows(df):
    """코인 약어는 비었는데 다른 칸을 적은 줄의 번호(저장하면 사라지므로 미리 알려준다)."""
    if df is None or df.empty:
        return []
    out = []
    for i, r in df.reset_index(drop=True).iterrows():
        sym = norm_symbol(r.get("코인"))
        other = any(str(r.get(c) if r.get(c) is not None else "").strip().upper() not in _BLANK for c in ("수량", "평균단가", "메모"))
        if sym == "" and other:
            out.append(i + 1)
    return out


def fetch_prices(symbols, get=None):
    """코인 약어 -> 원화 현재가. 반환: (가격 dict, 출처 dict, 시세를 못 가져온 약어 목록, 안내 목록)"""
    get = get or requests.get
    syms = sorted({norm_symbol(s) for s in symbols if norm_symbol(s)})
    prices, src, notes = {}, {}, []
    if not syms:
        return prices, src, [], notes
    try:                                                              # 업비트: 상장된 원화 마켓만 골라서 한 번에 묻는다(없는 마켓이 섞이면 전체가 실패해서)
        r = get(UPBIT_ALL, params={"isDetails": "false"}, headers=HEADERS, timeout=10)
        r.raise_for_status()
        listed = {m["market"] for m in r.json() if str(m.get("market", "")).startswith("KRW-")}
        want = [f"KRW-{s}" for s in syms if f"KRW-{s}" in listed]
        if want:
            r = get(UPBIT_TICKER, params={"markets": ",".join(want)}, headers=HEADERS, timeout=10)
            r.raise_for_status()
            for t in r.json():
                sym = str(t["market"])[4:]
                prices[sym], src[sym] = float(t["trade_price"]), "업비트"
    except Exception as e:
        notes.append(f"업비트 시세를 못 가져왔어요({str(e)[:60]})")
    for s in syms:                                                    # 업비트에 없으면 빗썸
        if s in prices:
            continue
        try:
            r = get(BITHUMB.format(sym=s), headers=HEADERS, timeout=10)
            r.raise_for_status()
            j = r.json()
            if str(j.get("status")) == "0000":
                prices[s], src[s] = float(j["data"]["closing_price"]), "빗썸"
        except Exception:
            pass
    missing = [s for s in syms if s not in prices]
    return prices, src, missing, notes


def values(df, prices):
    """평가금액·수익을 계산한 표. 시세가 없는 코인은 현재가와 평가금액이 비어 있다(합계에서 빠진다)."""
    d = clean(df)
    d["현재가"] = d["코인"].map(prices)
    d["평가금액"] = d["수량"] * d["현재가"]
    d["투자원금"] = d["수량"] * d["평균단가"]
    d["수익금액"] = d["평가금액"] - d["투자원금"]
    d["손익률"] = np.where(d["투자원금"] > 0, (d["평가금액"] / d["투자원금"] - 1) * 100, np.nan)
    return d
