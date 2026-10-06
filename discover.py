"""종목 발굴: 시장 전체에서 후보를 거르는 깔때기.
  1단계  종목 목록 필터(시가총액·거래대금·우선주/스팩 제외)
  2단계  DART 주요계정 일괄 조회(100개씩) -> 흑자·부채비율·ROE·PER·PBR 필터
  3단계  상위 후보만 상세 점검(안전 기준·가격 흐름) -> 앱에서 처리
가져오지 못한 값은 NaN으로 두고, 조용히 통과시키지 않는다."""
import datetime as dt

import numpy as np
import pandas as pd

import dart_data as dd

FIN_WORDS = ("금융", "은행", "증권", "보험", "캐피탈", "카드", "화재", "생명", "지주")  # 금융업 의심 이름
EXCLUDE_NAMES = ("스팩", "SPAC", "기업인수목적")

DEFAULT_FILTERS = {
    "cap": {"on": True, "v": 1.0},        # 시가총액(조원) 이상
    "tv": {"on": True, "v": 30.0},        # 일 거래대금(억원) 이상
    "profit": {"on": True},               # 최근 영업이익 흑자
    "loss": {"on": True, "v": 1},         # 최근 3년 영업적자 N번 이하
    "debt": {"on": True, "v": 100.0},     # 부채비율(%) 이하
    "roe": {"on": True, "v": 8.0},        # ROE(%) 이상
    "per": {"on": True, "v": 15.0},       # PER 이하
    "pbr": {"on": False, "v": 1.5},       # PBR 이하
    "growth": {"on": False},              # 영업이익이 전년보다 증가
    "maxn": 400,                          # 2단계에서 조회할 최대 종목 수(시가총액 순)
}


# ---------- 종목 목록 ----------
def load_listing():
    """FinanceDataReader로 상장 종목 목록(시가총액·거래대금 포함)을 가져온다. 실패하면 예외."""
    import FinanceDataReader as fdr
    last = None
    for mk in ("KRX",):
        try:
            df = fdr.StockListing(mk)
            if df is not None and len(df):
                return normalize_listing(df)
        except Exception as e:
            last = e
    raise RuntimeError(f"종목 목록을 가져오지 못했어요: {last}")


def normalize_listing(df):
    d = df.copy()
    if "Code" not in d.columns and "Symbol" in d.columns:
        d = d.rename(columns={"Symbol": "Code"})
    if "Code" not in d.columns or "Name" not in d.columns:
        raise RuntimeError("종목 목록의 형식이 예상과 달라요(Code/Name 열이 없어요).")
    for c in ("Marcap", "Amount", "Close"):
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "Market" not in d.columns:
        d["Market"] = ""
    if "Dept" not in d.columns:
        d["Dept"] = ""
    d["Dept"] = d["Dept"].fillna("").astype(str)
    d["Code"] = d["Code"].astype(str).str.strip().str.upper().str.zfill(6)
    d["Name"] = d["Name"].astype(str)
    return d[["Code", "Name", "Market", "Dept", "Close", "Marcap", "Amount"]].drop_duplicates("Code").reset_index(drop=True)


def manual_listing(codes):
    codes = [str(c).strip().upper().zfill(6) for c in codes if str(c).strip()]
    return pd.DataFrame({"Code": list(dict.fromkeys(codes)), "Name": "", "Market": "", "Dept": "", "Close": np.nan, "Marcap": np.nan, "Amount": np.nan})


def load_admin_codes(timeout=20):
    """관리종목으로 지정된 종목코드 집합. 실패하거나 오래 걸리면 예외."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

    def _go():
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX-ADMINISTRATIVE")
        return set(df["Symbol"].astype(str).str.strip().str.zfill(6))

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(_go).result(timeout=timeout)
    except FTimeout:
        raise RuntimeError(f"{timeout}초 안에 응답이 없었어요")
    finally:
        ex.shutdown(wait=False)


def is_common_stock(code, name):
    if not code or code[-1] != "0":
        return False  # 우선주 등(끝자리 5·7·9·K·L·M)은 제외
    return not any(w in name for w in EXCLUDE_NAMES)


# ---------- 1단계 ----------
def stage1(uni, f, exclude=()):
    d = uni.copy()
    d = d[[is_common_stock(c, n) if n else c[-1:] == "0" for c, n in zip(d["Code"], d["Name"])]]
    if exclude:
        d = d[~d["Code"].isin(set(exclude))]
    has_cap = d["Marcap"].notna().any()
    notes = []
    if f["cap"]["on"]:
        if has_cap:
            d = d[d["Marcap"] >= f["cap"]["v"] * 1e12]
        else:
            notes.append("시가총액 정보가 없어서 시가총액 조건은 적용하지 못했어요.")
    if f["tv"]["on"]:
        if d["Amount"].notna().any():
            d = d[d["Amount"] >= f["tv"]["v"] * 1e8]
        else:
            notes.append("거래대금 정보가 없어서 거래대금 조건은 적용하지 못했어요.")
    d = d.sort_values("Marcap", ascending=False, na_position="last")
    return d.head(int(f.get("maxn", 400))).reset_index(drop=True), notes


# ---------- 2단계: DART 주요계정 일괄 조회 ----------
ACCOUNTS = {
    "op": ("영업이익", "영업이익(손실)", "영업손익"),
    "ni": ("당기순이익", "당기순이익(손실)", "당기순손익", "연결당기순이익"),
    "equity": ("자본총계",),
    "liab": ("부채총계",),
    "rev": ("매출액", "수익(매출액)", "영업수익"),
}


def parse_multi(rows):
    """fnlttMultiAcnt 응답 -> {종목코드: {op:[전전기,전기,당기], ni, equity, liab, rev, fs}}"""
    by = {}
    for r in rows:
        by.setdefault(str(r.get("stock_code", "")).zfill(6), []).append(r)
    out = {}
    for code, rs in by.items():
        use = [r for r in rs if r.get("fs_div") == "CFS"] or [r for r in rs if r.get("fs_div") == "OFS"] or rs
        rec = {"fs": use[0].get("fs_div", "")}
        for key, names in ACCOUNTS.items():
            row = next((r for r in use if dd._norm(r.get("account_nm", "")) in {dd._norm(n) for n in names}), None)
            if row is None:
                rec[key] = None
                continue
            rec[key] = [dd._num(row.get("bfefrmtrm_amount")), dd._num(row.get("frmtrm_amount")), dd._num(row.get("thstrm_amount"))]
        out[code] = rec
    return out


def bulk_financials(client, codes, today=None, progress=None):
    """종목코드 목록의 최근 사업보고서 주요계정을 100개씩 묶어서 조회한다. 반환: (결과 dict, 오류 목록)"""
    today = today or dt.date.today()
    c2c = client.corp_map()
    result, errors = {}, []
    for year in (today.year - 1, today.year - 2):
        pending = [c for c in codes if c not in result and c in c2c]
        batches = [pending[i:i + 100] for i in range(0, len(pending), 100)]
        for bi, batch in enumerate(batches):
            if progress:
                progress(f"{year}년 사업보고서 조회 중… ({bi + 1}/{len(batches)})")
            try:
                jo = dd._call(client.api_key, "fnlttMultiAcnt",
                              {"corp_code": ",".join(c2c[c] for c in batch), "bsns_year": year, "reprt_code": "11011"})
            except Exception as e:
                errors.append(f"{year}년 묶음 {bi + 1}: {str(e)[:120]}")
                if any(k in str(e) for k in ("010", "011", "020")):
                    return result, errors
                continue
            for code, rec in parse_multi(jo.get("list", [])).items():
                if code in batch:
                    rec["year"] = year
                    result[code] = rec
    return result, errors


def metrics_table(stage1_df, fin):
    rows = []
    for r in stage1_df.itertuples():
        f = fin.get(r.Code)
        rec = {"종목코드": r.Code, "종목명": r.Name, "시가총액(조)": r.Marcap / 1e12 if pd.notna(r.Marcap) else np.nan,
               "거래대금(억)": r.Amount / 1e8 if pd.notna(r.Amount) else np.nan, "현재가": r.Close, "재무": f is not None}
        op = (f or {}).get("op")
        known = [x for x in (op or []) if x is not None]
        rec["영업이익"] = op[2] if op and op[2] is not None else np.nan
        rec["영업적자횟수"] = sum(1 for x in known if x <= 0) if known else np.nan
        rec["영업이익증가율"] = ((op[2] / op[1] - 1) * 100) if op and op[1] and op[1] > 0 and op[2] is not None else np.nan
        ni = ((f or {}).get("ni") or [None, None, None])[2]
        eq = ((f or {}).get("equity") or [None, None, None])[2]
        li = ((f or {}).get("liab") or [None, None, None])[2]
        rv = ((f or {}).get("rev") or [None, None, None])[2]
        rec["ROE"] = ni / eq * 100 if (ni is not None and eq and eq > 0) else np.nan
        rec["부채비율"] = li / eq * 100 if (li is not None and eq and eq > 0) else np.nan
        rec["PER"] = (r.Marcap / ni) if (pd.notna(r.Marcap) and ni and ni > 0) else np.nan
        rec["PBR"] = (r.Marcap / eq) if (pd.notna(r.Marcap) and eq and eq > 0) else np.nan
        rec["영업이익률"] = (op[2] / rv * 100) if (op and op[2] is not None and rv and rv > 0) else np.nan
        rec["금융업의심"] = any(w in (r.Name or "") for w in FIN_WORDS)
        rec["재무연도"] = (f or {}).get("year")
        rows.append(rec)
    return pd.DataFrame(rows)


def stage2(tbl, f, has_cap=True):
    """재무 조건. 값이 없으면(NaN) 탈락시킨다. 금융업 의심 종목은 부채비율 조건을 면제한다."""
    d = tbl.copy()
    ok = d["재무"].copy()
    if f["profit"]["on"]:
        ok &= d["영업이익"] > 0
    if f["loss"]["on"]:
        ok &= d["영업적자횟수"] <= f["loss"]["v"]
    if f["debt"]["on"]:
        ok &= (d["부채비율"] <= f["debt"]["v"]) | (d["금융업의심"] & d["재무"])
    if f["roe"]["on"]:
        ok &= d["ROE"] >= f["roe"]["v"]
    if has_cap and f["per"]["on"]:
        ok &= d["PER"] <= f["per"]["v"]
    if has_cap and f["pbr"]["on"]:
        ok &= d["PBR"] <= f["pbr"]["v"]
    if f["growth"]["on"]:
        ok &= d["영업이익증가율"] > 0
    d["통과"] = ok.fillna(False)
    return d


SORTS = {
    "ROE 높은 순": ("ROE", False),
    "PER 낮은 순": ("PER", True),
    "PBR 낮은 순": ("PBR", True),
    "영업이익 증가율 높은 순": ("영업이익증가율", False),
    "영업이익률 높은 순": ("영업이익률", False),
    "시가총액 큰 순": ("시가총액(조)", False),
}
