"""개별 증권사 리포트 목록(증권사·날짜·투자의견·목표가). 네이버 금융 리서치 목록과 리포트 상세 페이지의 머리 정보만 읽는다.
리포트 본문은 가져오지 않는다. 형식이 바뀌면 읽지 못한 항목은 비워 두고 이유를 알려준다."""
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

LIST_URL = "https://finance.naver.com/research/company_list.naver"
READ_URL = "https://finance.naver.com/research/company_read.naver"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://finance.naver.com/research/",
}


class ReportError(Exception):
    pass


def _num(x):
    try:
        s = re.sub(r"[^\d.]", "", str(x))
        return float(s) if s else None
    except Exception:
        return None


def _soup(html):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser")


def _date(s):
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{2})", s.strip())
    return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def describe(html):
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.S | re.I)
    title = re.sub(r"\s+", " ", m.group(1)).strip()[:40] if m else "제목 없음"
    return f"응답 {len(html or ''):,}자, 제목 '{title}', 표 {len(re.findall('<table', html or '', re.I))}개"


def parse_list(html):
    """리서치 목록 HTML -> [{nid, title, firm, date, url}] (최신순)"""
    soup = _soup(html)
    rows = []
    for tr in soup.find_all("tr"):
        a = tr.find("a", href=re.compile(r"company_read\.naver"))
        tds = tr.find_all("td")
        if not a or len(tds) < 4:
            continue
        nid = re.search(r"nid=(\d+)", a.get("href", ""))
        texts = [td.get_text(" ", strip=True) for td in tds]
        idx = next((i for i, td in enumerate(tds) if a in td.find_all("a")), None)
        date = next((_date(t) for t in texts if _date(t)), None)
        firm = texts[idx + 1] if idx is not None and idx + 1 < len(texts) else ""
        if not nid or not date:
            continue
        rows.append({"nid": nid.group(1), "title": a.get_text(" ", strip=True), "firm": firm, "date": date,
                     "url": f"{READ_URL}?nid={nid.group(1)}"})
    if not rows:
        raise ReportError(f"리포트 목록을 찾지 못했어요({describe(html)}).")
    return rows


def parse_read(html):
    """리포트 상세 페이지 -> {'opinion': 문자열 또는 None, 'target': 숫자 또는 None}"""
    soup = _soup(html)
    tgt = soup.select_one("em.money strong") or soup.select_one("em.money")
    op = soup.select_one("em.coment")
    target = _num(tgt.get_text()) if tgt else None
    opinion = op.get_text(strip=True) if op else None
    if target is None or not opinion:
        text = soup.get_text(" ", strip=True)
        if target is None:
            m = re.search(r"목표가\s*[:：]?\s*([\d,]{3,})", text)
            target = _num(m.group(1)) if m else None
        if not opinion:
            m = re.search(r"투자의견\s*[:：]?\s*([가-힣A-Za-z]{2,12})", text)
            opinion = m.group(1) if m else None
    return {"opinion": opinion, "target": target}


def _get(get, url, params):
    try:
        r = get(url, params=params, headers=HEADERS, timeout=12)
    except Exception as e:
        raise ReportError(f"접속 실패({str(e)[:60]})")
    if getattr(r, "status_code", 200) != 200:
        raise ReportError(f"HTTP {r.status_code}")
    r.encoding = "euc-kr"
    return r.text


def fetch_reports(code, n=12, get=None):
    """최근 리포트 n건. 반환: (rows, notes) — rows는 최신순, notes는 못 읽은 항목 안내."""
    get = get or requests.get
    code = str(code).zfill(6)
    rows = []
    for page in (1, 2):
        html = _get(get, LIST_URL, {"searchType": "itemCode", "itemCode": code, "page": page})
        try:
            rows += parse_list(html)
        except ReportError:
            if page == 1:
                raise
            break
        if len(rows) >= n:
            break
    rows = rows[:n]

    def detail(r):
        try:
            return parse_read(_get(get, READ_URL, {"nid": r["nid"]}))
        except Exception as e:
            return {"opinion": None, "target": None, "err": str(e)[:60]}

    with ThreadPoolExecutor(max_workers=4) as ex:
        for r, d in zip(rows, ex.map(detail, rows)):
            r["opinion"], r["target"] = d["opinion"], d["target"]
            r["err"] = d.get("err")
    # 같은 증권사의 바로 앞 목표가와 비교
    for i, r in enumerate(rows):
        r["prev"] = next((q["target"] for q in rows[i + 1:] if q["firm"] == r["firm"] and q["target"]), None)
    notes = []
    missing = sum(1 for r in rows if r["target"] is None)
    if missing:
        errs = {r["err"] for r in rows if r.get("err")}
        notes.append(f"{len(rows)}건 중 {missing}건은 의견·목표가를 읽지 못했어요" + (f"({', '.join(sorted(errs))[:80]})" if errs else "(페이지 형식이 달라서일 수 있어요)") + ".")
    return rows, notes


def summarize(rows, price=None):
    t = [r["target"] for r in rows if r["target"]]
    ups = sum(1 for r in rows if r["target"] and r["prev"] and r["target"] > r["prev"])
    downs = sum(1 for r in rows if r["target"] and r["prev"] and r["target"] < r["prev"])
    ops = [r["opinion"] for r in rows if r["opinion"]]
    buy = sum(1 for o in ops if any(k in o for k in ("매수", "Buy", "BUY", "Outperform", "비중확대")))
    return {"n": len(rows), "n_target": len(t), "avg": float(np.mean(t)) if t else None, "hi": max(t) if t else None, "lo": min(t) if t else None,
            "ups": ups, "downs": downs, "buy": buy, "n_op": len(ops),
            "upside": ((np.mean(t) / price - 1) * 100) if (t and price) else None}
