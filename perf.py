"""성과 계산: 자산 기록(스냅샷)과 입출금으로 입금 보정 수익률·고점 대비 낙폭·월말 자산을 만든다.
총자산 = 주식 평가금액 + 현금. 입금·출금이 수익률로 둔갑하지 않게, 기간별 수익률을 이어 붙인다(시간가중)."""
import numpy as np
import pandas as pd

SNAP_COLS = ["날짜", "주식", "현금", "총자산", "구분", "메모"]
FLOW_COLS = ["날짜", "금액", "메모"]
DEFAULT_GOAL = {"min": 3.0, "max": 8.0, "dd": 15.0, "kakao_flow": True, "inflation": 2.5, "infl_on": True}


def clean_snaps(df):
    d = df.copy()
    for c in SNAP_COLS:
        if c not in d.columns:
            d[c] = np.nan
    d["날짜"] = pd.to_datetime(d["날짜"], errors="coerce")
    for c in ("주식", "현금", "총자산"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False).replace({"": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")
    calc = d["주식"].fillna(0) + d["현금"].fillna(0)
    d["총자산"] = d["총자산"].where(d["총자산"].notna(), calc.where((d["주식"].notna()) | (d["현금"].notna())))
    d = d.dropna(subset=["날짜", "총자산"])
    d = d[d["총자산"] > 0]
    return d.sort_values("날짜").drop_duplicates("날짜", keep="last").reset_index(drop=True)


def clean_flows(df):
    d = df.copy()
    for c in FLOW_COLS:
        if c not in d.columns:
            d[c] = np.nan
    d["날짜"] = pd.to_datetime(d["날짜"], errors="coerce")
    d["금액"] = pd.to_numeric(d["금액"].astype(str).str.replace(",", "", regex=False).replace({"": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")
    return d.dropna(subset=["날짜", "금액"]).sort_values("날짜").reset_index(drop=True)


def kakao_start(r, today=None):
    """정기매수 한 줄의 시작일. 횟수와 기준일이 있으면 기준일 기준 (횟수-1) 영업일 전, 없으면 시작일."""
    cnt = pd.to_numeric(r.get("횟수"), errors="coerce")
    if pd.notna(cnt):
        a = pd.to_datetime(r.get("기준일"), errors="coerce")
        a = (today or pd.Timestamp.today().normalize()) if pd.isna(a) else a
        c = int(cnt)
        if c >= 1:
            return pd.bdate_range(end=a, periods=c)[0]
        return a + pd.offsets.BDay(1)
    return pd.to_datetime(r.get("시작일"), errors="coerce")


def kakao_flows(kakao_df, start, end):
    """카카오 정기매수를 매 영업일 외부 입금으로 본다. start 다음날부터 end까지."""
    rows = []
    if kakao_df is not None and len(kakao_df):
        for r in kakao_df.to_dict("records"):
            s = kakao_start(r)
            amt = pd.to_numeric(r.get("하루금액"), errors="coerce")
            if pd.isna(s) or pd.isna(amt) or amt <= 0:
                continue
            first = max(s, start + pd.Timedelta(days=1))
            if first > end:
                continue
            rows += [(d, float(amt), "카카오 정기매수") for d in pd.bdate_range(first, end)]
    return pd.DataFrame(rows, columns=FLOW_COLS) if rows else pd.DataFrame(columns=FLOW_COLS)


def all_flows(flows, kakao_df, use_kakao, start, end):
    f = clean_flows(flows)
    if use_kakao:
        k = kakao_flows(kakao_df, start, end)
        if len(k):
            f = pd.concat([f, k], ignore_index=True)
    return f.sort_values("날짜").reset_index(drop=True) if len(f) else f


def build_series(snaps, flows):
    """기록 사이마다 (끝값 - 시작값 - 입금) / (시작값 + 입금/2)를 구하고 이어 붙여 지수를 만든다."""
    rows, idx, prev = [], 1.0, None
    for r in snaps.itertuples():
        if prev is None:
            rows.append({"날짜": r.날짜, "총자산": r.총자산, "입출금": 0.0, "구간수익률": np.nan, "지수": 1.0})
        else:
            m = (flows["날짜"] > prev.날짜) & (flows["날짜"] <= r.날짜) if len(flows) else None
            f = float(flows.loc[m, "금액"].sum()) if m is not None else 0.0
            base = prev.총자산 + 0.5 * f
            ret = (r.총자산 - prev.총자산 - f) / base if base > 0 else 0.0
            idx *= 1 + ret
            rows.append({"날짜": r.날짜, "총자산": r.총자산, "입출금": f, "구간수익률": ret, "지수": idx})
        prev = r
    return pd.DataFrame(rows)


def year_stats(series, year):
    """올해 수익률. 연초 기준은 작년 말 이전의 마지막 기록, 없으면 올해 첫 기록."""
    if series is None or len(series) < 2:
        return None
    ys = pd.Timestamp(year=year, month=1, day=1)
    before = series[series["날짜"] < ys]
    base = before.iloc[-1] if len(before) else series[series["날짜"] >= ys].iloc[0]
    end = series.iloc[-1]
    if end["날짜"] <= base["날짜"]:
        return None
    seg = series[(series["날짜"] > base["날짜"]) & (series["날짜"] <= end["날짜"])]
    flow = float(seg["입출금"].sum())
    twr = end["지수"] / base["지수"] - 1
    return {
        "base_date": base["날짜"], "end_date": end["날짜"], "base_total": float(base["총자산"]), "end_total": float(end["총자산"]),
        "flow": flow, "profit": float(end["총자산"] - base["총자산"] - flow), "twr": float(twr),
        "simple": float(end["총자산"] / base["총자산"] - 1), "approx_base": len(before) == 0,
    }


def drawdown(series, since=None):
    d = series if since is None else series[series["날짜"] >= since]
    if d is None or len(d) < 1:
        return None
    peak = d["지수"].cummax()
    dd = d["지수"] / peak - 1
    return {"series": pd.DataFrame({"날짜": d["날짜"].values, "낙폭": dd.values}), "current": float(dd.iloc[-1]), "max": float(dd.min()),
            "peak_date": d.loc[d["지수"].idxmax(), "날짜"]}


def month_end(series):
    if series is None or not len(series):
        return pd.DataFrame(columns=["월", "월말 총자산", "월간 수익률", "입출금"])
    s = series.copy()
    s["월"] = s["날짜"].dt.to_period("M")
    rows = []
    for m, g in s.groupby("월"):
        rets = g["구간수익률"].dropna()
        rows.append({"월": str(m), "월말 총자산": float(g["총자산"].iloc[-1]),
                     "월간 수익률": float((1 + rets).prod() - 1) if len(rets) else np.nan, "입출금": float(g["입출금"].sum())})
    return pd.DataFrame(rows)
