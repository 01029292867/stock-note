"""증권사 컨센서스(평균 목표주가·투자의견). 네이버 증권 모바일 데이터에서 읽는다.
리포트 원문이 아니라 요약 숫자만 쓴다. 형식이 바뀌거나 접속이 막히면 ConsensusError로 이유를 알려준다."""
import numpy as np
import requests

HOSTS = ["https://m.stock.naver.com/api/stock/{code}/integration", "https://api.stock.naver.com/stock/{code}/integration"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36",
    "Referer": "https://m.stock.naver.com/",
}
KNOWN = ("stockName", "itemCode", "totalInfos", "dealTrendInfos", "consensusInfo", "researches", "industryCompareInfo")


class ConsensusError(Exception):
    pass


def _num(x):
    try:
        s = str(x).replace(",", "").replace("원", "").strip()
        return float(s) if s not in ("", "-", "None", "nan") else None
    except Exception:
        return None


def parse_integration(data):
    """integration JSON -> {'has': bool, 'target': 원, 'score': 1~5, 'date': 기준일}"""
    if not isinstance(data, dict) or not any(k in data for k in KNOWN):
        keys = ", ".join(list(data)[:12]) if isinstance(data, dict) else type(data).__name__
        raise ConsensusError(f"응답 형식이 달라요(항목: {keys[:120]})")
    ci = data.get("consensusInfo")
    if not isinstance(ci, dict):
        return {"has": False, "target": None, "score": None, "date": None}
    t, s = _num(ci.get("priceTargetMean")), _num(ci.get("recommMean"))
    if t is None and s is None:
        return {"has": False, "target": None, "score": None, "date": None}
    return {"has": True, "target": t, "score": s, "date": str(ci.get("createDate") or "")[:10]}


def fetch(code, get=None):
    get = get or requests.get
    errors = []
    for url in HOSTS:
        host = url.split("/")[2]
        try:
            r = get(url.format(code=str(code).zfill(6)), headers=HEADERS, timeout=12)
        except Exception as e:
            errors.append(f"{host}: 접속 실패({str(e)[:50]})")
            continue
        if getattr(r, "status_code", 200) != 200:
            errors.append(f"{host}: HTTP {r.status_code}")
            continue
        try:
            data = r.json()
        except Exception:
            errors.append(f"{host}: JSON이 아니에요(응답 {len(getattr(r, 'text', '') or ''):,}자)")
            continue
        try:
            return parse_integration(data)
        except ConsensusError as e:
            errors.append(f"{host}: {e}")
    raise ConsensusError(" / ".join(errors))


def label(score):
    """네이버 기준 5점 만점(5 적극매수 ~ 1 적극매도)"""
    if score is None:
        return "-"
    return "적극매수" if score >= 4.5 else "매수" if score >= 3.5 else "중립" if score >= 2.5 else "매도" if score >= 1.5 else "적극매도"


def upside(target, price):
    return (target / price - 1) * 100 if target and price else np.nan
