"""종목별 업종(섹터) 정보. 네이버 금융의 업종 페이지를 먼저 읽고, 안 되면 FinanceDataReader의 KRX 업종을 쓴다.
형식을 이 환경에서 직접 확인하지 못해서, 읽지 못하면 이유를 알려주고 업종 없이 진행한다."""
VERSION = 1

import re

import requests

INDEX = "https://finance.naver.com/sise/sise_group.naver"
DETAIL = "https://finance.naver.com/sise/sise_group_detail.naver"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Referer": "https://finance.naver.com/sise/"}


class SectorError(Exception):
    pass


IDX_RE = re.compile(r'sise_group_detail\.(?:naver|nhn)\?type=upjong(?:&amp;|&)no=(\d+)[^>]*>\s*([^<]+?)\s*<', re.S)
CODE_RE = re.compile(r'main\.(?:naver|nhn)\?code=(\d{6})')


def parse_index(html):
    """업종 목록 페이지 -> [(번호, 업종 이름)]. 링크 형식(.naver/.nhn, &amp;)이 달라도 읽는다."""
    out, seen = [], set()
    for no, name in IDX_RE.findall(html or ""):
        name = re.sub(r"\s+", " ", name).strip()
        if name and no not in seen:
            seen.add(no)
            out.append((no, name))
    if not out:
        hrefs = re.findall(r'href=["\']([^"\']*(?:upjong|sise_group)[^"\']*)["\']', html or "")
        raise SectorError(f"업종 목록을 찾지 못했어요({len(html or ''):,}자, 관련 링크 {len(hrefs)}개" + (f", 예: {hrefs[0][:90]}" if hrefs else "") + ")")
    return out


def parse_detail(html):
    """업종 상세 페이지 -> 소속 종목코드 목록"""
    html = html or ""
    m = re.search(r'<table[^>]*class="[^"]*type_5[^"]*"[^>]*>.*?</table>', html, re.S)
    scope = m.group(0) if m else html
    codes = []
    for c in CODE_RE.findall(scope):
        if c not in codes:
            codes.append(c)
    return codes


def from_naver(get=None, workers=8):
    from concurrent.futures import ThreadPoolExecutor
    get = get or requests.get

    def fetch(url, params):
        r = get(url, params=params, headers=HEADERS, timeout=12)
        if getattr(r, "status_code", 200) != 200:
            raise SectorError(f"HTTP {r.status_code}")
        r.encoding = "euc-kr"
        return r.text

    idx = parse_index(fetch(INDEX, {"type": "upjong"}))
    mp, errs = {}, []

    def one(item):
        no, name = item
        try:
            return name, parse_detail(fetch(DETAIL, {"type": "upjong", "no": no}))
        except Exception as e:
            return name, str(e)[:60]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, res in ex.map(one, idx):
            if isinstance(res, list):
                for c in res:
                    mp.setdefault(c, name)
            elif len(errs) < 3:
                errs.append(f"{name}: {res}")
    if not mp:
        raise SectorError("업종별 종목을 읽지 못했어요: " + "; ".join(errs))
    return mp


def from_fdr():
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX-DESC")
    col = "Sector" if "Sector" in df.columns else ("Industry" if "Industry" in df.columns else None)
    if col is None or "Code" not in df.columns:
        raise SectorError("KRX 업종 열이 없어요")
    d = df[["Code", col]].dropna()
    return {str(c).zfill(6): str(s) for c, s in zip(d["Code"], d[col]) if str(s).strip()}


def load():
    """반환: ({종목코드: 업종}, 출처 이름, 안내 문구). 모두 실패하면 ({}, '', 이유)."""
    errs = []
    for name, fn in (("네이버 금융 업종", from_naver), ("KRX 업종", from_fdr)):
        try:
            mp = fn()
            if mp:
                return mp, name, "; ".join(errs)
        except Exception as e:
            errs.append(f"{name}: {str(e)[:80]}")
    return {}, "", "업종 정보를 가져오지 못했어요 — " + "; ".join(errs)
