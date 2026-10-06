"""DART(전자공시) 재무·공시 수집. 가져오지 못한 항목은 None(확인 불가)으로 남긴다.
오류와 '데이터 없음'을 구분해서, 통신 실패가 '문제없음'으로 둔갑하지 않게 한다."""
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "https://opendart.fss.or.kr/api/"
EVENT_KEYS = {  # 이름: DART 엔드포인트
    "유상증자": "piicDecsn",
    "전환사채": "cvbdIsDecsn",
    "신주인수권부사채": "bdwtIsDecsn",
    "교환사채": "exbdIsDecsn",
}
DISTRESS_KEYS = {"부도발생": "dfOcr", "회생절차": "ctrcvsBgrq", "관리절차개시": "bnkMngtPcbg"}


class DartError(Exception):
    pass


def _num(x):
    try:
        s = str(x).replace(",", "").strip()
        if s in ("", "-", "nan", "None"):
            return None
        return float(s)
    except Exception:
        return None


def _norm(s):
    return str(s).replace(" ", "").replace("\u3000", "")


def _call(api_key, endpoint, params, timeout=25):
    """성공이면 JSON dict, '조회된 데이터 없음(013)'이면 빈 list, 그 외는 DartError."""
    r = requests.get(BASE + endpoint + ".json", params=dict(params, crtfc_key=api_key), timeout=timeout)
    r.raise_for_status()
    jo = r.json()
    status = str(jo.get("status", ""))
    if status == "000":
        return jo
    if status == "013":
        return {"status": "013", "list": []}
    raise DartError(f"DART 응답 {status}: {jo.get('message', '')}")


class DartClient:
    def __init__(self, api_key):
        import OpenDartReader  # 이 라이브러리는 import한 이름 자체가 클래스예요
        self.api_key = api_key
        self._r = OpenDartReader(api_key)

    def corp_code(self, stock_code):
        c = self._r.find_corp_code(str(stock_code).zfill(6))
        if not c:
            raise DartError("DART에서 이 종목코드의 회사를 찾지 못했어요(상장사가 맞는지 확인하세요).")
        return c


# ---------- 항목별 수집 ----------
def _pick(rows, sjs, ids, names):
    cand = [r for r in rows if r.get("sj_div") in sjs]
    for i in ids:
        for r in cand:
            if r.get("account_id") == i:
                return r
    for n in names:
        for r in cand:
            if _norm(r.get("account_nm", "")) == _norm(n):
                return r
    return None


def _amounts(row):
    """[전전기, 전기, 당기] 순서(오래된 것부터)."""
    if row is None:
        return None
    return [_num(row.get("bfefrmtrm_amount")), _num(row.get("frmtrm_amount")), _num(row.get("thstrm_amount"))]


def parse_financials(rows):
    op = _pick(rows, ("IS", "CIS"), ["dart_OperatingIncomeLoss"], ["영업이익", "영업이익(손실)", "영업손익"])
    ni = _pick(rows, ("IS", "CIS"), ["ifrs-full_ProfitLoss", "ifrs_ProfitLoss"], ["당기순이익", "당기순이익(손실)", "당기순손익", "연결당기순이익"])
    eq = _pick(rows, ("BS",), ["ifrs-full_Equity", "ifrs_Equity"], ["자본총계"])
    li = _pick(rows, ("BS",), ["ifrs-full_Liabilities", "ifrs_Liabilities"], ["부채총계"])
    cap = _pick(rows, ("BS",), ["ifrs-full_IssuedCapital", "ifrs_IssuedCapital"], ["자본금"])
    ocf = _pick(rows, ("CF",), ["ifrs-full_CashFlowsFromUsedInOperatingActivities", "ifrs_CashFlowsFromUsedInOperatingActivities"],
                ["영업활동현금흐름", "영업활동으로인한현금흐름", "영업활동순현금흐름"])
    intr = _pick(rows, ("IS", "CIS"), ["ifrs-full_InterestExpense"], ["이자비용"])
    basis = "이자비용"
    if intr is None:
        intr = _pick(rows, ("IS", "CIS"), ["ifrs-full_FinanceCosts"], ["금융비용", "금융원가", "금융비용(수익)"])
        basis = "금융비용(이자 외 항목 포함)"
    latest = lambda r: (_amounts(r) or [None, None, None])[2]
    return {
        "op": _amounts(op), "ni": latest(ni), "equity": latest(eq), "liab": latest(li),
        "capital": latest(cap), "ocf": latest(ocf),
        "interest": abs(latest(intr)) if latest(intr) is not None else None,
        "interest_basis": basis if intr is not None else None,
    }


def _fetch_fin(client, corp, ty):
    last_err = None
    for y in (ty - 1, ty - 2):
        for fs in ("CFS", "OFS"):
            jo = _call(client.api_key, "fnlttSinglAcntAll", {"corp_code": corp, "bsns_year": y, "reprt_code": "11011", "fs_div": fs})
            rows = jo.get("list", [])
            if rows:
                d = parse_financials(rows)
                d.update({"year": y, "fs": fs})
                return d
    return None


def _report_latest(client, endpoint, corp, ty):
    for y in (ty - 1, ty - 2):
        rows = _call(client.api_key, endpoint, {"corp_code": corp, "bsns_year": y, "reprt_code": "11011"}).get("list", [])
        if rows:
            return y, rows
    return None, []


def _fetch_div(client, corp, ty):
    y, rows = _report_latest(client, "alotMatter", corp, ty)
    if not rows:
        return None
    cand = [r for r in rows if "주당" in str(r.get("se", "")) and "현금배당" in str(r.get("se", ""))]
    pref = [r for r in cand if "보통" in str(r.get("stock_knd", ""))] or cand
    if not pref:
        return None
    r = pref[0]
    vals = [_num(r.get("lwfr")), _num(r.get("frmtrm")), _num(r.get("thstrm"))]
    # '-'는 배당 없음(0)으로 본다. 행 자체가 없으면 위에서 None 처리했다.
    return {"year": y, "per_share": [v if v is not None else 0.0 for v in vals]}


def _fetch_major(client, corp, ty):
    y, rows = _report_latest(client, "hyslrSttus", corp, ty)
    if not rows:
        return None
    tot = [r for r in rows if _norm(r.get("nm", "")) == "계"]
    tot_c = [r for r in tot if "보통" in str(r.get("stock_knd", ""))] or tot
    if tot_c:
        v = _num(tot_c[0].get("trmend_posesn_stock_qota_rt"))
        if v is not None:
            return {"year": y, "ratio": v}
    s = [_num(r.get("trmend_posesn_stock_qota_rt")) for r in rows if "보통" in str(r.get("stock_knd", ""))]
    s = [x for x in s if x is not None]
    return {"year": y, "ratio": sum(s)} if s else None


def _fetch_shares(client, corp, ty):
    y, rows = _report_latest(client, "stockTotqySttus", corp, ty)
    for r in rows:
        if "보통" in str(r.get("se", "")):
            v = _num(r.get("istc_totqy"))
            if v:
                return {"year": y, "common": v}
    return None


def _fetch_audit(client, corp, ty):
    y, rows = _report_latest(client, "accnutAdtorNmNdAdtOpinion", corp, ty)
    if not rows:
        return None
    op = str(rows[0].get("adt_opinion", "")).strip()
    return {"year": y, "opinion": op, "auditor": str(rows[0].get("adtor", ""))}


def _count_events(client, corp, keys, start, end):
    out = {}
    for name, ep in keys.items():
        rows = _call(client.api_key, ep, {"corp_code": corp, "bgn_de": start, "end_de": end}).get("list", [])
        out[name] = len(rows)
    return out


def _fetch_company(client, corp):
    jo = _call(client.api_key, "company", {"corp_code": corp})
    return {"name": jo.get("corp_name", ""), "induty": str(jo.get("induty_code", ""))}


def fetch_all(client, stock_code, today=None):
    """한 종목의 DART 데이터를 모은다. 반환값은 JSON 저장이 가능한 dict."""
    today = today or dt.date.today()
    ty = today.year
    corp = client.corp_code(stock_code)
    start = (today.replace(year=today.year - 3)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    tasks = {
        "company": lambda: _fetch_company(client, corp),
        "fin": lambda: _fetch_fin(client, corp, ty),
        "div": lambda: _fetch_div(client, corp, ty),
        "major": lambda: _fetch_major(client, corp, ty),
        "shares": lambda: _fetch_shares(client, corp, ty),
        "audit": lambda: _fetch_audit(client, corp, ty),
        "issues": lambda: _count_events(client, corp, EVENT_KEYS, start, end),
        "distress": lambda: _count_events(client, corp, DISTRESS_KEYS, start, end),
    }
    result = {"code": str(stock_code).zfill(6), "fetched": today.isoformat(), "errors": {}}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {k: ex.submit(f) for k, f in tasks.items()}
        for k, f in futs.items():
            try:
                result[k] = f.result()
            except Exception as e:
                result[k] = None
                result["errors"][k] = str(e)[:200]
    return result
