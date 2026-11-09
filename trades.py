"""거래 기록(원장): 매수·매도·배당을 남기고, 매도의 확정 손익과 거래 성적을 계산한다.
평단은 이동평균 방식(보유 종목의 평균단가)이라 증권사 화면의 평단과 약간 다를 수 있다. 기록한 날부터 쌓인다(예전 거래는 복원하지 못한다)."""
VERSION = 1

import datetime as dt

import numpy as np
import pandas as pd

import measure

TRADE_COLS = ["id", "날짜", "증권사", "종목코드", "종목명", "구분", "수량", "단가", "금액", "수수료세금", "평단", "실현손익", "실현수익률", "이유", "꼬리표", "메모"]
KINDS = ["매도", "배당", "매수"]
PLAN_REASONS = ["계획대로 정리(목표 도달)", "무효가격 이탈", "기간 만료"]
_NUM = ["수량", "단가", "금액", "수수료세금", "평단", "실현손익", "실현수익률"]
_TXT = ["id", "증권사", "종목코드", "종목명", "구분", "이유", "꼬리표", "메모"]


def reason_group(r):
    """판단 기록과 같은 기준으로 이유를 묶는다(계획대로 정리·무효가격 이탈·기간 만료는 규칙 기반)."""
    if str(r) in PLAN_REASONS:
        return "근거·규칙 기반"
    return measure.reason_group(r)


def clean(df):
    d = df.copy() if df is not None else pd.DataFrame(columns=TRADE_COLS)
    for c in TRADE_COLS:
        if c not in d.columns:
            d[c] = np.nan
    d = d[TRADE_COLS].copy()
    d["날짜"] = pd.to_datetime(d["날짜"], errors="coerce")
    for c in _NUM:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False).replace({"": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan}), errors="coerce")
    for c in _TXT:
        d[c] = d[c].fillna("").astype(str).replace({"None": "", "nan": ""})
    d["종목코드"] = d["종목코드"].str.strip().str.replace(r"\.0$", "", regex=True).map(lambda s: s.zfill(6) if s.isdigit() else s)
    d = d[d["날짜"].notna() & d["구분"].isin(KINDS)]
    return d.sort_values(["날짜", "id"], kind="stable").reset_index(drop=True)


def realized(price, avg, qty, cost=0.0):
    """확정 손익(원)과 수익률(%): (매도가 − 평단) × 수량 − 수수료·세금, 수익률은 매입 원가(평단 × 수량) 대비."""
    basis = float(avg) * float(qty)
    pnl = (float(price) - float(avg)) * float(qty) - float(cost or 0)
    return pnl, (pnl / basis * 100 if basis > 0 else float("nan"))


def _find(h, broker, code):
    m = (h["증권사"].astype(str) == str(broker)) & (h["종목코드"].astype(str) == str(code))
    return h.index[m][0] if m.any() else None


def apply_buy(hold, broker, code, name, tag, qty, price, cost, date):
    """매수를 보유 종목에 반영한다(이동평균 평단: (기존 수량 × 기존 평단 + 산 금액 + 비용) ÷ 새 수량). 새 종목이면 줄을 추가한다."""
    h = hold.copy().reset_index(drop=True)
    qty, price, cost = round(float(qty), 8), float(price), float(cost or 0)
    if qty <= 0 or price <= 0:
        raise ValueError("수량과 단가는 0보다 커야 해요.")
    i = _find(h, broker, code)
    d = pd.Timestamp(date).date().isoformat()
    if i is not None:
        q0, a0 = float(pd.to_numeric(h.at[i, "수량"], errors="coerce") or 0), float(pd.to_numeric(h.at[i, "평균단가"], errors="coerce") or 0)
        q1 = q0 + qty
        h.at[i, "수량"], h.at[i, "평균단가"] = round(q1, 8), round((q0 * a0 + qty * price + cost) / q1, 4)
        if not str(h.at[i, "매수일"] if "매수일" in h.columns else "").strip() or str(h.at[i, "매수일"]) in ("None", "nan"):
            h.at[i, "매수일"] = d
        if not str(h.at[i, "종목명"]).strip() and name:
            h.at[i, "종목명"] = name
        return h
    row = {"증권사": broker, "종목코드": code, "종목명": name, "꼬리표": tag, "수량": qty, "평균단가": round((qty * price + cost) / qty, 4), "매수일": d}
    return pd.concat([h, pd.DataFrame([row])], ignore_index=True)


def apply_sell(hold, broker, code, qty):
    """매도를 보유 종목에서 뺀다. 전량이면 줄을 지운다. 반환: (새 보유 종목, 매도 시점 평단, 줄이 지워졌는지)"""
    h = hold.copy().reset_index(drop=True)
    i = _find(h, broker, code)
    if i is None:
        raise ValueError("보유 종목에서 그 종목(증권사)을 찾지 못했어요.")
    q0 = float(pd.to_numeric(h.at[i, "수량"], errors="coerce") or 0)
    qty = round(float(qty), 8)
    if qty <= 0:
        raise ValueError("수량은 0보다 커야 해요.")
    if qty > q0 + 1e-9:
        raise ValueError(f"보유 수량({q0:g})보다 많이 팔 수 없어요.")
    avg = float(pd.to_numeric(h.at[i, "평균단가"], errors="coerce") or 0)
    left = round(q0 - qty, 8)
    if left <= 1e-9:
        return h.drop(index=i).reset_index(drop=True), avg, True
    h.at[i, "수량"] = left
    return h, avg, False


def make_row(kind, date, broker, code, name, qty=None, price=None, amount=None, cost=0.0, avg=None, reason="", tag="", memo=""):
    """거래 한 줄. 매도는 확정 손익을, 배당은 금액만 담는다."""
    row = {"id": dt.datetime.now().strftime("%Y%m%d%H%M%S%f")[:20] + str(code), "날짜": pd.Timestamp(date).date().isoformat(), "증권사": broker, "종목코드": code, "종목명": name, "구분": kind,
           "수량": qty, "단가": price, "금액": amount, "수수료세금": cost or 0.0, "평단": avg, "실현손익": np.nan, "실현수익률": np.nan, "이유": reason, "꼬리표": tag, "메모": memo}
    if kind in ("매도", "매수") and qty is not None and price is not None:
        row["금액"] = round(float(qty) * float(price), 2)
    if kind == "매도":
        pnl, pct = realized(price, avg, qty, cost)
        row["실현손익"], row["실현수익률"] = round(pnl, 2), round(pct, 4) if pct == pct else np.nan
    return row


def _between(d, start, end):
    if start is not None:
        d = d[d["날짜"] >= pd.Timestamp(start)]
    if end is not None:
        d = d[d["날짜"] <= pd.Timestamp(end)]
    return d


def stats(df, start=None, end=None):
    """기간 안의 거래 성적: 매도 건수, 승률, 평균 이익·손실, 손익비, 건당 기대값, 실현손익 합계, 배당 합계."""
    d = _between(clean(df), start, end)
    s, dv = d[d["구분"] == "매도"], d[d["구분"] == "배당"]
    out = {"n": int(len(s)), "div": float(dv["금액"].sum()), "total": float(s["실현손익"].sum())}
    out["net"] = out["total"] + out["div"]
    if len(s):
        w, l = s[s["실현손익"] > 0], s[s["실현손익"] < 0]
        out.update({"wins": int(len(w)), "losses": int(len(l)), "win_rate": len(w) / len(s) * 100, "avg_win": float(w["실현손익"].mean()) if len(w) else np.nan,
                    "avg_loss": float(l["실현손익"].mean()) if len(l) else np.nan, "expectancy": float(s["실현손익"].mean()), "exp_pct": float(s["실현수익률"].mean()),
                    "best": float(s["실현손익"].max()), "worst": float(s["실현손익"].min())})
        out["payoff"] = out["avg_win"] / abs(out["avg_loss"]) if (len(w) and len(l) and out["avg_loss"]) else np.nan
    return out


def monthly(df):
    d = clean(df)
    d = d[d["구분"].isin(["매도", "배당"])]
    if d.empty:
        return pd.DataFrame(columns=["월", "실현손익", "배당", "합계", "매도 건수"])
    d = d.assign(월=d["날짜"].dt.strftime("%Y-%m"))
    rows = []
    for m, g in d.groupby("월"):
        p, v = float(g.loc[g["구분"] == "매도", "실현손익"].sum()), float(g.loc[g["구분"] == "배당", "금액"].sum())
        rows.append({"월": m, "실현손익": p, "배당": v, "합계": p + v, "매도 건수": int((g["구분"] == "매도").sum())})
    return pd.DataFrame(rows)


def by_stock(df, start=None, end=None):
    d = _between(clean(df), start, end)
    d = d[d["구분"].isin(["매도", "배당"])]
    if d.empty:
        return pd.DataFrame(columns=["종목명", "매도 건수", "실현손익", "승률(%)", "배당"])
    rows = []
    for k, g in d.groupby(d["종목명"].where(d["종목명"] != "", d["종목코드"])):
        s = g[g["구분"] == "매도"]
        rows.append({"종목명": k, "매도 건수": int(len(s)), "실현손익": float(s["실현손익"].sum()), "승률(%)": float((s["실현손익"] > 0).mean() * 100) if len(s) else np.nan, "배당": float(g.loc[g["구분"] == "배당", "금액"].sum())})
    return pd.DataFrame(rows).sort_values("실현손익", ascending=False).reset_index(drop=True)


def by_reason(df, start=None, end=None):
    """매도 이유 묶음(근거·규칙 / 느낌·감정·남의 말 / 기타)별 성적."""
    d = _between(clean(df), start, end)
    s = d[d["구분"] == "매도"].copy()
    if s.empty:
        return pd.DataFrame(columns=["묶음", "건수", "승률(%)", "평균 수익률(%)", "실현손익", "참고"])
    s["묶음"] = s["이유"].map(reason_group)
    rows = []
    for k, g in s.groupby("묶음"):
        rows.append({"묶음": k, "건수": int(len(g)), "승률(%)": float((g["실현손익"] > 0).mean() * 100), "평균 수익률(%)": float(g["실현수익률"].mean()), "실현손익": float(g["실현손익"].sum()),
                     "참고": "표본 적음(참고만)" if len(g) < 20 else ""})
    return pd.DataFrame(rows)
