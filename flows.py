"""투자자별 수급(외국인·기관·개인). 네이버 금융 종목별 외국인·기관 매매 페이지를 읽는다(스크래핑).
개인은 따로 제공되지 않아서 '-(기관+외국인)'로 추정한다(기타법인 등이 섞여 있어요).
페이지 형식이 바뀌거나 접속이 막히면 FlowError로 이유를 알려준다."""
import io
import time

import numpy as np
import pandas as pd
import requests

URL = "https://finance.naver.com/item/frgn.naver"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://finance.naver.com/",
}


class FlowError(Exception):
    pass


def _num(x):
    try:
        s = str(x).replace(",", "").replace("%", "").replace("+", "").strip()
        if s in ("", "-", "nan", "None"):
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def describe_html(html):
    """실패했을 때 네이버가 돌려준 페이지가 어떤 모양인지 한 줄로 알려준다."""
    import re
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.S | re.I)
    title = re.sub(r"\s+", " ", m.group(1)).strip()[:40] if m else "제목 없음"
    return f"응답 {len(html or ''):,}자, 제목 '{title}', 표 {len(re.findall('<table', html or '', re.I))}개"


def parse_frgn_html(html):
    """네이버 외국인·기관 매매 페이지 HTML -> 날짜 오름차순 DataFrame"""
    try:
        tables = pd.read_html(io.StringIO(html), match="기관")
    except ValueError:
        raise FlowError(f"수급 표를 찾지 못했어요({describe_html(html)}).")
    except ImportError as e:
        raise FlowError(f"표를 읽는 부품이 설치되어 있지 않아요: {str(e)[:80]}")
    except Exception as e:
        raise FlowError(f"수급 표를 읽다가 오류가 났어요: {str(e)[:100]}")
    for t in tables:
        cols = ["|".join(dict.fromkeys(str(x) for x in c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        joined = " ".join(cols)
        if "날짜" in joined and "기관" in joined and "외국인" in joined:
            break
    else:
        raise FlowError(f"날짜·기관·외국인 열이 있는 표를 찾지 못했어요({describe_html(html)}).")
    t = t.copy()
    t.columns = cols

    def find(*keys, excl=()):
        for c in cols:
            if all(k in c for k in keys) and not any(e in c for e in excl):
                return c
        return None

    c_date, c_close, c_inst = find("날짜"), find("종가"), find("기관")
    c_frgn = find("외국인", "순매매량") or find("외국인", excl=("보유",))
    c_vol, c_hold, c_pct = find("거래량"), find("외국인", "보유주수"), find("보유율")
    if not all([c_date, c_close, c_inst, c_frgn]):
        raise FlowError(f"필요한 열을 찾지 못했어요(찾은 열: {', '.join(cols)}).")
    d = pd.DataFrame({
        "날짜": pd.to_datetime(t[c_date].astype(str).str.strip(), format="%Y.%m.%d", errors="coerce"),
        "종가": t[c_close].map(_num),
        "거래량": t[c_vol].map(_num) if c_vol else np.nan,
        "기관": t[c_inst].map(_num),
        "외국인": t[c_frgn].map(_num),
        "외국인보유주수": t[c_hold].map(_num) if c_hold else np.nan,
        "외국인보유율": t[c_pct].map(_num) if c_pct else np.nan,
    })
    d = d.dropna(subset=["날짜", "종가"]).sort_values("날짜")
    if d.empty:
        raise FlowError("수급 표는 찾았지만 읽을 수 있는 행이 없었어요.")
    return d.drop_duplicates("날짜").reset_index(drop=True)


MOBILE = "https://m.stock.naver.com/api/stock/{code}/trend"
KEYS = {"date": ("bizdate", "localTradedAt", "date"), "close": ("closePrice",), "frgn": ("foreignerPureBuyQuant",),
        "inst": ("organPureBuyQuant",), "indiv": ("individualPureBuyQuant",), "pct": ("foreignerHoldRatio",),
        "vol": ("accumulatedTradingVolume",)}


def parse_trend_json(data):
    """네이버 모바일 trend JSON -> 날짜 오름차순 DataFrame(개인 포함)"""
    rows = data
    if isinstance(data, dict):
        rows = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise FlowError("모바일 API 응답에 목록이 없어요.")
    keys = set().union(*[set(r) for r in rows[:3]])

    def pick(name):
        return next((k for k in KEYS[name] if k in keys), None)

    kd, kc, kf, ki = pick("date"), pick("close"), pick("frgn"), pick("inst")
    if not all([kd, kc, kf, ki]):
        raise FlowError(f"모바일 API 응답의 항목 이름이 달라요(찾은 항목: {', '.join(sorted(keys))[:150]}).")
    kp, kh, kv = pick("indiv"), pick("pct"), pick("vol")

    def to_date(x):
        s = str(x).strip()
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce") if s.isdigit() and len(s) == 8 else pd.to_datetime(s, errors="coerce")

    d = pd.DataFrame({
        "날짜": [to_date(r.get(kd)) for r in rows], "종가": [_num(r.get(kc)) for r in rows],
        "거래량": [_num(r.get(kv)) if kv else np.nan for r in rows],
        "기관": [_num(r.get(ki)) for r in rows], "외국인": [_num(r.get(kf)) for r in rows],
        "개인": [_num(r.get(kp)) if kp else np.nan for r in rows],
        "외국인보유주수": np.nan, "외국인보유율": [_num(r.get(kh)) if kh else np.nan for r in rows]})
    d = d.dropna(subset=["날짜", "종가"]).sort_values("날짜")
    if d.empty:
        raise FlowError("모바일 API에서 읽을 수 있는 행이 없었어요.")
    return d.drop_duplicates("날짜").reset_index(drop=True)


def _fetch_json(code, pages, get, pause):
    try:
        r = get(MOBILE.format(code=str(code).zfill(6)), params={"pageSize": pages * 20},
                headers=dict(HEADERS, Referer="https://m.stock.naver.com/"), timeout=12)
    except Exception as e:
        raise FlowError(f"접속 실패({str(e)[:60]})")
    if getattr(r, "status_code", 200) != 200:
        raise FlowError(f"HTTP {r.status_code}")
    try:
        data = r.json()
    except Exception:
        raise FlowError(f"JSON이 아니에요(응답 {len(getattr(r, 'text', '') or ''):,}자)")
    return parse_trend_json(data)


def _fetch_html(code, pages, get, pause):
    frames = []
    for p in range(1, pages + 1):
        try:
            r = get(URL, params={"code": str(code).zfill(6), "page": p}, headers=HEADERS, timeout=12)
        except Exception as e:
            raise FlowError(f"접속에 실패했어요: {str(e)[:100]}")
        if getattr(r, "status_code", 200) != 200:
            raise FlowError(f"HTTP {r.status_code}로 거절됐어요(서버에서 접속이 막혔을 수 있어요).")
        r.encoding = "euc-kr"
        frames.append(parse_frgn_html(r.text))
        if p < pages:
            time.sleep(pause)
    return pd.concat(frames, ignore_index=True).drop_duplicates("날짜").sort_values("날짜").reset_index(drop=True)


def fetch_flow(code, pages=2, get=None, pause=0.3):
    """한 종목의 최근 수급(약 pages*20거래일). 모바일 API를 먼저, 안 되면 웹 페이지를 읽는다."""
    get = get or requests.get
    errors = []
    for name, fn in (("모바일 API", _fetch_json), ("웹 페이지", _fetch_html)):
        try:
            return fn(code, pages, get, pause)
        except FlowError as e:
            errors.append(f"{name}: {e}")
    raise FlowError(" / ".join(errors))


def with_amounts(df):
    """순매매량(주)을 종가를 곱해 억원으로 바꾸고, 개인(추정)을 만든다."""
    d = df.sort_values("날짜").copy()
    d["기관(억)"] = d["기관"] * d["종가"] / 1e8
    d["외국인(억)"] = d["외국인"] * d["종가"] / 1e8
    est = -(d["기관(억)"] + d["외국인(억)"])
    if "개인" in d.columns and d["개인"].notna().any():
        d["개인(억)"] = d["개인"] * d["종가"] / 1e8
        d["개인추정"] = False
    else:
        d["개인(억)"] = est
        d["개인추정"] = True
    return d


def _streak(s):
    """맨 끝에서부터 같은 방향(순매수 +/순매도 -)이 며칠 이어졌는지. 부호 있는 정수."""
    vals = [v for v in s.tolist() if v == v]
    if not vals or vals[-1] == 0:
        return 0
    sign = 1 if vals[-1] > 0 else -1
    n = 0
    for v in reversed(vals):
        if v * sign > 0:
            n += 1
        else:
            break
    return sign * n


def summarize(df):
    d = with_amounts(df)
    if len(d) < 5:
        return None
    s = lambda col, n: float(d[col].tail(n).sum())
    out = {"last": d["날짜"].iloc[-1].date().isoformat(), "n": int(len(d)),
           "f5": s("외국인(억)", 5), "i5": s("기관(억)", 5), "p5": s("개인(억)", 5),
           "f20": s("외국인(억)", 20), "i20": s("기관(억)", 20), "p20": s("개인(억)", 20),
           "f_streak": _streak(d["외국인(억)"]), "i_streak": _streak(d["기관(억)"]),
           "indiv_est": bool(d["개인추정"].iloc[-1])}
    pct = d["외국인보유율"].dropna()
    out["hold_pct"] = float(pct.iloc[-1]) if len(pct) else None
    out["hold_chg20"] = float(pct.iloc[-1] - pct.iloc[-min(20, len(pct))]) if len(pct) >= 2 else None
    return out


def read_text(sm):
    """수급을 한 문단으로 풀어쓴다. 판단이 아니라 현재 흐름 설명이다."""
    f5, i5, p5, f20, i20 = sm["f5"], sm["i5"], sm["p5"], sm["f20"], sm["i20"]
    if f5 > 0 and i5 > 0:
        main = "최근 5일 외국인과 기관이 함께 순매수했어요" + (", 개인은 순매도 쪽이에요." if p5 < 0 else ".")
    elif f5 < 0 and i5 < 0:
        main = "최근 5일 외국인과 기관이 함께 순매도했어요" + (", 개인이 받아 사는 쪽이에요." if p5 > 0 else ".")
    elif f5 > 0:
        main = "최근 5일 외국인은 순매수, 기관은 순매도로 방향이 엇갈려요."
    else:
        main = "최근 5일 기관은 순매수, 외국인은 순매도로 방향이 엇갈려요."
    turn = ""
    if f20 < 0 < f5:
        turn = " 외국인은 20일 기준 순매도였는데 최근 5일은 순매수로 돌아섰어요."
    elif f20 > 0 > f5:
        turn = " 외국인은 20일 기준 순매수였는데 최근 5일은 순매도로 돌아섰어요."
    elif i20 < 0 < i5:
        turn = " 기관은 20일 기준 순매도였는데 최근 5일은 순매수로 돌아섰어요."
    elif i20 > 0 > i5:
        turn = " 기관은 20일 기준 순매수였는데 최근 5일은 순매도로 돌아섰어요."
    return main + turn


def chart_frame(df, n=20):
    d = with_amounts(df).tail(n)
    rows = []
    ind = "개인(추정)" if bool(d["개인추정"].iloc[-1]) else "개인"
    for name, col in (("외국인", "외국인(억)"), ("기관", "기관(억)"), (ind, "개인(억)")):
        rows += [{"날짜": a, "주체": name, "순매수(억)": b} for a, b in zip(d["날짜"], d[col])]
    return pd.DataFrame(rows)
