"""코인 평가(보기 전용): 거래소, 코인 약어, 수량, 평균단가를 직접 적으면 원화 시세로 평가금액을 계산한다.
시세는 업비트 공개 API(없는 코인은 빗썸)에서 가져오며 API 키는 필요 없다. 주문은 하지 않는다."""
VERSION = 2  # 2: 수량 소수점, 매수금액으로 수량·평균단가 계산

import numpy as np
import pandas as pd
import requests

COLS = ["거래소", "코인", "수량", "평균단가", "메모"]
EDIT_COLS = COLS + ["매수금액"]          # 편집 표에만 있는 칸: 수량이나 평균단가를 모를 때 계산하는 데 쓰고, 저장하면 비워진다
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


def _num(x):
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else float("nan")


def derive(qty, avg, amt):
    """수량·평균단가·매수금액 중 두 가지를 알면 나머지 하나를 계산한다. 반환: (수량, 평균단가)."""
    qty, avg, amt = _num(qty), _num(avg), _num(amt)
    if not (qty > 0) and avg > 0 and amt > 0:
        qty = round(amt / avg, 8)
    elif qty > 0 and not (avg > 0) and amt > 0:
        avg = round(amt / qty, 6)
    return (qty if qty > 0 else 0.0), (avg if avg > 0 else float("nan"))


def clean(df):
    """입력 표를 정리한다: 코인 약어를 대문자로, 수량·평균단가를 숫자로(소수점 가능). 코인 약어가 없는 줄은 뺀다.
    매수금액 칸이 있으면 수량이나 평균단가를 모를 때 거기서 계산해 채운다(매수금액 칸은 결과에 남기지 않는다)."""
    d = df.copy() if df is not None else pd.DataFrame(columns=EDIT_COLS)
    for c in EDIT_COLS:
        if c not in d.columns:
            d[c] = None
    d = d[EDIT_COLS].copy()
    d["코인"] = d["코인"].map(norm_symbol)
    d = d[d["코인"] != ""].reset_index(drop=True)
    q, a = [], []
    for r in d.itertuples(index=False):
        qq, aa = derive(r.수량, r.평균단가, r.매수금액)
        q.append(qq)
        a.append(aa)
    d["수량"], d["평균단가"] = pd.Series(q, dtype="float64"), pd.Series(a, dtype="float64")
    d["거래소"] = d["거래소"].fillna("").astype(str).replace({"None": "", "nan": ""}).where(lambda s: s != "", "기타")
    d["메모"] = d["메모"].fillna("").astype(str).replace({"None": "", "nan": ""})
    return d[COLS].reset_index(drop=True)


def blank_symbol_rows(df):
    """코인 약어는 비었는데 다른 칸을 적은 줄의 번호(저장하면 사라지므로 미리 알려준다)."""
    if df is None or df.empty:
        return []
    out = []
    for i, r in df.reset_index(drop=True).iterrows():
        sym = norm_symbol(r.get("코인"))
        other = any(str(r.get(c) if r.get(c) is not None else "").strip().upper() not in _BLANK for c in ("수량", "평균단가", "메모", "매수금액"))
        if sym == "" and other:
            out.append(i + 1)
    return out


def unknown_qty_rows(df):
    """코인 약어는 있는데 수량을 알 수 없는 줄의 번호(수량을 적거나, 매수금액과 평균단가를 같이 적어야 한다)."""
    if df is None or df.empty:
        return []
    d = df.reset_index(drop=True)
    out = []
    for i, r in d.iterrows():
        if norm_symbol(r.get("코인")) and not (derive(r.get("수량"), r.get("평균단가"), r.get("매수금액"))[0] > 0):
            out.append(i + 1)
    return out


def fmt_price(x):
    """원화 가격을 보기 좋게: 큰 값은 천 단위 쉼표, 작은 값(수십 원 이하)은 소수점 몇 자리까지."""
    v = _num(x)
    if v != v:
        return ""
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


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
