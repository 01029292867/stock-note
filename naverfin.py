"""네이버 금융에서 읽어오는 재무(DART가 막힐 때의 대체 자료).
가져올 수 있는 것: 매출액·영업이익·당기순이익(최근 연간), ROE, 부채비율, PER, PBR, 주당배당금.
가져올 수 없는 것(DART 공시에만 있는 것): 이자보상배율, 영업현금흐름, 자본잠식, 감사의견, 최대주주 지분, 증자·CB·BW 이력, 부도·회생 공시."""
VERSION = 2  # 2: 시가총액 순위(종목 목록의 시가총액·거래대금 보충)

import re

import requests

MOBILE = "https://m.stock.naver.com/api/stock/{code}/finance/annual"
WEB = "https://finance.naver.com/item/main.naver"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}
WANT = ("매출액", "영업이익", "당기순이익", "ROE", "부채비율", "PER", "PBR", "주당배당금", "EPS", "BPS")


class NaverFinError(Exception):
    pass


def _num(x):
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        if s in ("", "-", "N/A", "nan", "None"):
            return None
        return float(s)
    except Exception:
        return None


def _year(label):
    m = re.search(r"(20\d\d|19\d\d)", str(label))
    return int(m.group(1)) if m else None


def parse_json(data):
    """모바일 API 응답 -> {'labels', 'est', 'rows'}"""
    info = (data or {}).get("financeInfo") if isinstance(data, dict) else None
    if not info or not info.get("rowList") or not info.get("trTitleList"):
        keys = ", ".join(list(data)[:8]) if isinstance(data, dict) else type(data).__name__
        raise NaverFinError(f"모바일 API 응답 형식이 달라요(항목: {keys[:80]})")
    titles = info["trTitleList"]
    keys = [t.get("key") for t in titles]
    labels = [str(t.get("title", "")) for t in titles]
    est = [str(t.get("isConsensus", "N")).upper() == "Y" for t in titles]
    rows = {}
    for r in info["rowList"]:
        cols = r.get("columns") or {}
        rows[str(r.get("title", "")).strip()] = [_num((cols.get(k) or {}).get("value")) for k in keys]
    return {"labels": labels, "est": est, "rows": rows}


def parse_html(html):
    """종목 메인 페이지의 '기업실적분석' 표 -> {'labels', 'est', 'rows'} (최근 연간 실적 4개 열)"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    tb = soup.select_one("div.cop_analysis table")
    if tb is None:
        raise NaverFinError(f"기업실적분석 표를 찾지 못했어요({len(html or ''):,}자, 표 {len(soup.find_all('table'))}개)")
    heads = [tr for tr in tb.select("thead tr")]
    ths = []
    for tr in heads:
        cells = [th.get_text(" ", strip=True) for th in tr.find_all("th")]
        if sum(1 for c in cells if _year(c)) >= 3:
            ths = [c for c in cells if _year(c)]
            break
    if not ths:
        raise NaverFinError("기업실적분석 표의 연도 열을 찾지 못했어요")
    labels = ths[:4]
    est = ["(E)" in c or "E)" in c for c in labels]
    rows = {}
    for tr in tb.select("tbody tr"):
        th = tr.find("th")
        if not th:
            continue
        title = th.get_text(" ", strip=True)
        vals = [_num(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
        if vals:
            rows[title] = vals[:4]
    if not rows:
        raise NaverFinError("기업실적분석 표에서 읽을 행이 없었어요")
    return {"labels": labels, "est": est, "rows": rows}


def fetch_annual(code, get=None):
    """한 종목의 최근 연간 재무. 모바일 API를 먼저, 안 되면 웹 페이지."""
    get = get or requests.get
    code = str(code).zfill(6)
    errors = []
    try:
        r = get(MOBILE.format(code=code), headers=dict(HEADERS, Referer="https://m.stock.naver.com/"), timeout=12)
        if getattr(r, "status_code", 200) != 200:
            raise NaverFinError(f"HTTP {r.status_code}")
        return dict(parse_json(r.json()), source="모바일 API")
    except NaverFinError as e:
        errors.append(f"모바일 API: {e}")
    except Exception as e:
        errors.append(f"모바일 API: {type(e).__name__}")
    try:
        r = get(WEB, params={"code": code}, headers=HEADERS, timeout=12)
        if getattr(r, "status_code", 200) != 200:
            raise NaverFinError(f"HTTP {r.status_code}")
        r.encoding = "euc-kr"
        return dict(parse_html(r.text), source="웹 페이지")
    except NaverFinError as e:
        errors.append(f"웹 페이지: {e}")
    except Exception as e:
        errors.append(f"웹 페이지: {type(e).__name__}")
    raise NaverFinError(" / ".join(errors))


def _row(nf, name):
    for k, v in nf["rows"].items():
        if k.startswith(name):
            return v
    return None


def to_fin(nf):
    """discover가 쓰는 형태로 바꾼다. 금액은 원 단위, 비율은 %. 실적(E 아님)만 쓴다."""
    idx = [i for i, e in enumerate(nf["est"]) if not e]
    if not idx:
        raise NaverFinError("확정된 연간 실적이 없어요")
    use = idx[-3:]

    def amt(name):
        r = _row(nf, name)
        if r is None:
            return None
        vals = [(r[i] * 1e8 if (i < len(r) and r[i] is not None) else None) for i in use]
        while len(vals) < 3:
            vals.insert(0, None)
        return vals

    def last(name):
        r = _row(nf, name)
        if r is None:
            return None
        for i in reversed(idx):
            if i < len(r) and r[i] is not None:
                return r[i]
        return None

    div = _row(nf, "주당배당금")
    return {"op": amt("영업이익"), "ni": amt("당기순이익"), "rev": amt("매출액"), "equity": None, "liab": None,
            "roe": last("ROE"), "debt": last("부채비율"), "per": last("PER"), "pbr": last("PBR"),
            "div_list": ([div[i] for i in use if i < len(div)] if div else None),
            "year": _year(nf["labels"][use[-1]]), "fs": "NAVER"}


SUM = "https://finance.naver.com/sise/sise_market_sum.naver"


def parse_market_sum(html):
    """시가총액 순위 페이지 -> [{'Code','Name','Close','Marcap'(원),'Volume','Amount'(원, 현재가×거래량),'PER','ROE'}]"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    tb = soup.select_one("table.type_2")
    if tb is None:
        raise NaverFinError(f"시가총액 표를 찾지 못했어요({len(html or ''):,}자, 표 {len(soup.find_all('table'))}개)")
    heads = [th.get_text(" ", strip=True) for th in tb.select("thead th")]
    idx = {h: i for i, h in enumerate(heads)}

    def col(tds, name):
        i = idx.get(name)
        return _num(tds[i].get_text(" ", strip=True)) if i is not None and i < len(tds) else None

    out = []
    for tr in tb.select("tbody tr"):
        tds = tr.find_all("td")
        a = tr.find("a", href=re.compile(r"code=\d{6}"))
        if not a or len(tds) < 8:
            continue
        code = re.search(r"code=(\d{6})", a["href"]).group(1)
        price, cap, vol = col(tds, "현재가"), col(tds, "시가총액"), col(tds, "거래량")
        out.append({"Code": code, "Name": a.get_text(strip=True), "Close": price, "Marcap": cap * 1e8 if cap is not None else None,
                    "Volume": vol, "Amount": price * vol if (price is not None and vol is not None) else None,
                    "PER": col(tds, "PER"), "ROE": col(tds, "ROE")})
    if not out:
        raise NaverFinError("시가총액 표에서 읽을 행이 없었어요")
    return out


def market_top(n=600, get=None, pause=0.2):
    """네이버 금융 시가총액 순위에서 코스피·코스닥 상위 종목을 읽는다(시가총액 큰 순). 반환: DataFrame"""
    import time

    import pandas as pd
    get = get or requests.get
    rows, errs = [], []
    pages = int(n // 50) + 2
    for sosok in (0, 1):
        for page in range(1, pages + 1):
            try:
                r = get(SUM, params={"sosok": sosok, "page": page}, headers=HEADERS, timeout=12)
                if getattr(r, "status_code", 200) != 200:
                    raise NaverFinError(f"HTTP {r.status_code}")
                r.encoding = "euc-kr"
                rows += parse_market_sum(r.text)
            except Exception as e:
                errs.append(f"sosok{sosok} p{page}: {str(e)[:60]}")
                break
            time.sleep(pause)
    if not rows:
        raise NaverFinError("시가총액 순위를 읽지 못했어요: " + "; ".join(errs[:2]))
    df = pd.DataFrame(rows).drop_duplicates("Code")
    return df.sort_values("Marcap", ascending=False).head(n).reset_index(drop=True)
