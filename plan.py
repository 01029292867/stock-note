"""매도 계획 추적: 사기 전이나 보유 중에 정한 계획(무효가격, 1·2차 목표와 정리 비율, 보호선)을 현재 가격과 비교한다.
타이밍을 맞히는 게 아니라, 미리 정한 기준에 닿았을 때 점검하게 하는 도구다."""
VERSION = 1

import numpy as np
import pandas as pd

COLS = ["id", "종목코드", "종목명", "생성일", "상태", "꼬리표", "기준가", "수량", "무효가격", "목표1", "비율1", "목표2", "비율2", "보호폭", "완료1", "완료2", "메모"]


def clean(df):
    d = df.copy()
    for c in COLS:
        if c not in d.columns:
            d[c] = np.nan
    for c in ("기준가", "수량", "무효가격", "목표1", "비율1", "목표2", "비율2", "보호폭"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False).replace({"": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")
    d["종목코드"] = d["종목코드"].astype(str).str.strip().str.zfill(6)
    d["생성일"] = pd.to_datetime(d["생성일"], errors="coerce")
    for c in ("id", "종목명", "상태", "꼬리표", "완료1", "완료2", "메모"):
        d[c] = d[c].fillna("").astype(str)
    d.loc[d["상태"] == "", "상태"] = "진행"
    return d.dropna(subset=["생성일"])[COLS].reset_index(drop=True)


def evaluate(p, closes):
    """p: 계획 한 줄(dict). closes: 생성일 이후 종가 Series. 반환: 점검 결과 dict."""
    if closes is None or len(closes) == 0:
        return {"action": "시세를 가져오지 못했어요", "level": "info", "steps": [], "price": None}
    price = float(closes.iloc[-1])
    peak = float(closes.max())
    base = p["기준가"] if p["기준가"] == p["기준가"] else None
    inv, t1, t2, w = p["무효가격"], p["목표1"], p["목표2"], p["보호폭"]
    done1, done2 = p["완료1"] == "Y", p["완료2"] == "Y"
    steps = []

    def dist(x):
        return (x / price - 1) * 100 if x == x and x else np.nan

    inv_hit = inv == inv and bool(inv) and price <= inv
    t1_hit = t1 == t1 and bool(t1) and price >= t1
    t2_hit = t2 == t2 and bool(t2) and price >= t2
    t1_seen = t1 == t1 and bool(t1) and peak >= t1            # 한 번이라도 1차 목표를 넘었는지
    trail_line = (peak * (1 - w / 100)) if (w == w and w and t1_seen) else None
    trail_hit = trail_line is not None and price <= trail_line
    steps.append({"단계": "무효가격(전량 정리 점검)", "가격": inv, "현재가 대비(%)": dist(inv), "상태": "이탈" if inv_hit else "대기"})
    steps.append({"단계": f"1차 목표({p['비율1']:.0f}% 정리)" if p["비율1"] == p["비율1"] else "1차 목표", "가격": t1, "현재가 대비(%)": dist(t1),
                  "상태": "완료" if done1 else ("도달" if t1_hit else ("지나감" if t1_seen else "대기"))})
    steps.append({"단계": f"2차 목표({p['비율2']:.0f}% 정리)" if p["비율2"] == p["비율2"] else "2차 목표", "가격": t2, "현재가 대비(%)": dist(t2),
                  "상태": "완료" if done2 else ("도달" if t2_hit else "대기")})
    steps.append({"단계": f"보호선(고점 −{w:.0f}%, 1차 목표 이후)" if w == w and w else "보호선", "가격": trail_line, "현재가 대비(%)": dist(trail_line) if trail_line else np.nan,
                  "상태": "이탈" if trail_hit else ("추적 중" if trail_line else "아직 꺼짐")})
    if inv_hit:
        action, level = "무효가격 이탈 — 계획상 전량 정리를 점검할 시점이에요", "warn"
    elif trail_hit:
        action, level = "보호선 이탈 — 남은 수량 정리를 점검할 시점이에요", "warn"
    elif t2_hit and not done2:
        action, level = f"2차 목표 도달 — 계획상 {p['비율2']:.0f}% 정리를 점검할 시점이에요", "info"
    elif t1_hit and not done1:
        action, level = f"1차 목표 도달 — 계획상 {p['비율1']:.0f}% 정리를 점검할 시점이에요", "info"
    elif t1_seen and not done1:
        action, level = "1차 목표를 지나갔는데 아직 정리 완료로 표시하지 않았어요", "info"
    else:
        nxt = [s for s in steps[:3] if s["상태"] == "대기" and s["가격"] == s["가격"] and s["가격"]]
        nearest = min(nxt, key=lambda s: abs(s["현재가 대비(%)"])) if nxt else None
        action = f"계획 진행 중 — 가장 가까운 기준은 {nearest['단계']}({nearest['현재가 대비(%)']:+.1f}%)" if nearest else "계획 진행 중"
        level = "ok"
    return {"action": action, "level": level, "steps": steps, "price": price, "peak": peak, "trail_line": trail_line,
            "pl_pct": ((price / base - 1) * 100) if base else np.nan}
