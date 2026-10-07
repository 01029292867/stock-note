"""판단 기록과 결과 추적. 판단할 때의 숫자를 저장하고, 20·60·120거래일 뒤의 결과를 자동으로 붙여서
내 판단의 실제 승률·손익비·기대값을 보여준다. 표본이 적을 때는 참고만 하도록 표시한다."""
VERSION = 2  # 2: 기록 당시 종합점수 열 추가

import numpy as np
import pandas as pd

DECISIONS = ["매수", "추가 매수", "보유 유지", "일부 매도", "전량 매도", "관망(사지 않음)"]
HOLD_BUY = {"매수", "추가 매수", "보유 유지"}      # 이후 오르면 맞은 판단
REASONS = ["실적·재무", "저평가", "차트·추세", "수급", "증권사·뉴스", "공시·이벤트", "비중·리스크 관리", "불안·감정", "남의 추천", "느낌·우량주 판단", "월간 점검", "기타"]
HORIZONS = (20, 60, 120)
COLS = ["id", "날짜", "종목코드", "종목명", "결정", "이유", "확신도", "기록가", "평단", "수량", "꼬리표",
        "상승여력", "하락위험", "손익비", "등급", "필요승률", "무효가격", "메모", "종합점수"]
MIN_N = 20


def clean(df):
    d = df.copy()
    for c in COLS:
        if c not in d.columns:
            d[c] = np.nan
    for c in ("확신도", "기록가", "평단", "수량", "상승여력", "하락위험", "손익비", "필요승률", "무효가격", "종합점수"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False).replace({"": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")
    d["날짜"] = pd.to_datetime(d["날짜"], errors="coerce")
    d["종목코드"] = d["종목코드"].astype(str).str.strip().str.zfill(6)
    for c in ("종목명", "결정", "이유", "꼬리표", "등급", "메모", "id"):
        d[c] = d[c].fillna("").astype(str)
    return d.dropna(subset=["날짜"])[COLS].sort_values("날짜").reset_index(drop=True)


def direction(decision):
    """결정 손익의 방향: 사거나 들고 있는 결정은 오르면 +, 팔거나 사지 않은 결정은 내리면 +."""
    return 1 if decision in HOLD_BUY else -1


def evaluate(rec, closes_by_code, bench=None):
    """기록마다 horizons 뒤의 수익률(ret), 결정 손익(v), 코스피 대비(a), 구간 최대 하락(mae), 무효가격 이탈 여부를 계산한다."""
    rows = []
    for r in rec.to_dict("records"):
        s = closes_by_code.get(r["종목코드"])
        out = dict(r)
        out["현재까지(%)"] = np.nan
        for h in HORIZONS:
            for k in ("ret", "v", "a", "mae", "inv"):
                out[f"{k}{h}"] = np.nan
        if s is None or len(s) == 0 or not r["기록가"] or r["기록가"] != r["기록가"]:
            rows.append(out)
            continue
        idx = s.index
        i0 = int(idx.searchsorted(r["날짜"], side="right")) - 1
        if i0 < 0:
            rows.append(out)
            continue
        p0, sign = float(r["기록가"]), direction(r["결정"])
        out["현재까지(%)"] = (float(s.iloc[-1]) / p0 - 1) * 100
        out["경과일"] = len(s) - 1 - i0
        for h in HORIZONS:
            iN = i0 + h
            if iN >= len(s):
                continue
            seg = s.iloc[i0:iN + 1]
            ret = (float(s.iloc[iN]) / p0 - 1) * 100
            out[f"ret{h}"], out[f"v{h}"] = ret, sign * ret
            out[f"mae{h}"] = (float(seg.min()) / p0 - 1) * 100
            if r["무효가격"] == r["무효가격"] and r["무효가격"]:
                out[f"inv{h}"] = 1.0 if float(seg.min()) <= r["무효가격"] else 0.0
            if bench is not None and len(bench):
                b0 = bench.iloc[int(bench.index.searchsorted(idx[i0], side="right")) - 1] if bench.index.searchsorted(idx[i0], side="right") > 0 else np.nan
                bi = int(bench.index.searchsorted(idx[iN], side="right")) - 1
                if bi >= 0 and b0 == b0:
                    out[f"a{h}"] = sign * (ret - (float(bench.iloc[bi]) / float(b0) - 1) * 100)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize(ev, h=60, by=None):
    """결정 손익 v 기준 통계. by가 있으면 그 열로 나눠서, 없으면 결정 유형별과 전체를 보여준다."""
    col = f"v{h}"
    d = ev.dropna(subset=[col]) if len(ev) else ev
    if d is None or d.empty:
        return pd.DataFrame(columns=["구분", "건수", "맞은 비율(%)", "평균 이익(%)", "평균 손실(%)", "손익비", "건당 기대값(%)", "참고"])

    def row(name, g):
        v = g[col].astype(float)
        w, l = v[v > 0], v[v < 0]
        pay = (w.mean() / abs(l.mean())) if len(w) and len(l) else np.nan
        return {"구분": name, "건수": int(len(v)), "맞은 비율(%)": float((v > 0).mean() * 100),
                "평균 이익(%)": float(w.mean()) if len(w) else np.nan, "평균 손실(%)": float(l.mean()) if len(l) else np.nan,
                "손익비": float(pay) if pay == pay else np.nan, "건당 기대값(%)": float(v.mean()),
                "참고": "표본 적음(참고만)" if len(v) < MIN_N else ""}

    rows = [row("전체", d)]
    key = by or "결정"
    for k, g in d.groupby(key):
        rows.append(row(str(k), g))
    return pd.DataFrame(rows)


def confidence_bucket(x):
    return "높음(4~5)" if x >= 4 else "보통(3)" if x >= 3 else "낮음(1~2)" if x == x else "-"
