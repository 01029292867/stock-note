"""성과 확인: 이 앱이 실제로 도움이 되는지를 '미리 정한 기준'으로 잰다.
결과를 본 뒤에 기준을 바꾸면 우연이 실력처럼 보이므로, 프로토콜(평가 기간·최소 표본·처리 기한·목표)을 먼저 확정하고 그 기준으로만 판정한다."""
VERSION = 1

import re

import numpy as np
import pandas as pd

PROTOCOL_COLS = ["확정일", "시작일", "평가기간", "최소표본", "처리기한", "준수율목표", "메모"]
DEFAULT_PROTOCOL = {"평가기간": 60, "최소표본": 10, "처리기한": 3, "준수율목표": 80}

# 판단 기록의 '이유'를 두 묶음으로 나눠 비교한다(느낌·감정·남의 말로 한 결정이 근거·규칙으로 한 결정보다 나았나).
REASON_GROUP = {"실적·재무": "근거·규칙 기반", "저평가": "근거·규칙 기반", "차트·추세": "근거·규칙 기반", "수급": "근거·규칙 기반", "공시·이벤트": "근거·규칙 기반",
                "비중·리스크 관리": "근거·규칙 기반", "월간 점검": "근거·규칙 기반",
                "불안·감정": "느낌·감정·남의 말", "남의 추천": "느낌·감정·남의 말", "느낌·우량주 판단": "느낌·감정·남의 말", "증권사·뉴스": "느낌·감정·남의 말"}


def reason_group(r):
    return REASON_GROUP.get(str(r), "기타")


def clean_protocol(df):
    d = df.copy() if df is not None else pd.DataFrame(columns=PROTOCOL_COLS)
    for c in PROTOCOL_COLS:
        if c not in d.columns:
            d[c] = np.nan
    d["확정일"] = pd.to_datetime(d["확정일"], errors="coerce")
    d["시작일"] = pd.to_datetime(d["시작일"], errors="coerce")
    for c in ("평가기간", "최소표본", "처리기한", "준수율목표"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["메모"] = d["메모"].fillna("").astype(str).replace({"None": "", "nan": ""})
    return d.dropna(subset=["확정일", "시작일"])[PROTOCOL_COLS].sort_values("확정일").reset_index(drop=True)


def active_protocol(df):
    """가장 최근에 확정한 프로토콜(dict). 없으면 None."""
    d = clean_protocol(df)
    if d.empty:
        return None
    r = d.iloc[-1].to_dict()
    for k, v in DEFAULT_PROTOCOL.items():
        if r.get(k) != r.get(k) or not r.get(k):
            r[k] = v
    return r


def _asof(series, date):
    """date 이전(당일 포함) 마지막 값."""
    i = int(series.index.searchsorted(pd.Timestamp(date), side="right")) - 1
    return (float(series.iloc[i]), i) if i >= 0 else (np.nan, -1)


def watch_outcomes(watch, closes, bench, horizon=60, start=None):
    """관심종목(추가일·추가가가 있는 것)의 담은 뒤 성과를 코스피와 비교한다.
    경과일 >= horizon 인 것은 '평가 기간' 기준 수익과 초과수익도 계산한다(만기). start 이후에 담은 것만 프로토콜 표본이다."""
    rows = []
    for r in watch.to_dict("records"):
        d0 = pd.to_datetime(r.get("추가일"), errors="coerce")
        p0 = pd.to_numeric(r.get("추가가"), errors="coerce")
        s = closes.get(str(r.get("종목코드")).zfill(6))
        if pd.isna(d0) or pd.isna(p0) or p0 <= 0 or s is None or len(s) == 0:
            continue
        _, i0 = _asof(s, d0)
        if i0 < 0:
            i0 = 0
        b0, _ = _asof(bench, s.index[i0]) if bench is not None and len(bench) else (np.nan, -1)
        elapsed = len(s) - 1 - i0
        cur = (float(s.iloc[-1]) / p0 - 1) * 100
        bcur = (float(bench.iloc[-1]) / b0 - 1) * 100 if b0 == b0 and b0 else np.nan
        out = {"종목코드": str(r.get("종목코드")).zfill(6), "종목명": r.get("종목명") or r.get("종목코드"), "추가일": d0, "추가가": p0, "경과일": int(elapsed),
               "현재(%)": cur, "코스피(%)": bcur, "초과(%)": cur - bcur if bcur == bcur else np.nan,
               "만기": elapsed >= horizon, "기간 수익(%)": np.nan, "기간 코스피(%)": np.nan, "기간 초과(%)": np.nan,
               "표본": bool(start is None or d0 >= pd.Timestamp(start))}
        if elapsed >= horizon:
            iN = i0 + horizon
            ret = (float(s.iloc[iN]) / p0 - 1) * 100
            bN, _ = _asof(bench, s.index[iN]) if b0 == b0 and b0 else (np.nan, -1)
            br = (bN / b0 - 1) * 100 if bN == bN and b0 else np.nan
            out.update({"기간 수익(%)": ret, "기간 코스피(%)": br, "기간 초과(%)": ret - br if br == br else np.nan})
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_excess(x, n_boot=2000, seed=0):
    """초과수익 목록의 요약: 표본 수, 평균, 중앙값, 이긴 비율, 평균의 95% 신뢰구간(5개 이상일 때)."""
    a = np.asarray(pd.Series(x).dropna(), float)
    out = {"n": int(len(a))}
    if not len(a):
        return out
    out.update({"mean": float(a.mean()), "median": float(np.median(a)), "win": float((a > 0).mean() * 100)})
    if len(a) >= 5:
        rng = np.random.default_rng(seed)
        bs = [rng.choice(a, len(a)).mean() for _ in range(n_boot)]
        out["ci"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def judge_protocol(s, protocol):
    """미리 정한 성공 기준(평균 초과수익 > 0 이고 신뢰구간 하한 > 0, 표본 >= 최소 표본)으로 판정. 반환: (상태, 문구)"""
    need = int(protocol["최소표본"])
    if s.get("n", 0) < need:
        return "진행 중", f"표본 {s.get('n', 0)}/{need}개 — 아직 판단하지 않아요."
    lo, hi = s["ci"]
    if s["mean"] > 0 and lo > 0:
        return "충족", f"평균 초과수익 {s['mean']:+.1f}%p, 신뢰구간 {lo:+.1f} ~ {hi:+.1f}가 0보다 커서 **성공 기준을 충족**했어요."
    if s["mean"] < 0 and hi < 0:
        return "미충족", f"평균 초과수익 {s['mean']:+.1f}%p, 신뢰구간 {lo:+.1f} ~ {hi:+.1f}가 모두 0보다 작아서 **코스피에 뒤처졌어요.**"
    return "구분 안 됨", f"평균 초과수익 {s['mean']:+.1f}%p지만 신뢰구간 {lo:+.1f} ~ {hi:+.1f}가 0을 포함해서 **우연과 구분되지 않아요.**"


def entry_setup_rate(jr, start=None):
    """매수·추가 매수 기록 중 매수 전에 무효가격을 정해 둔 비율. 반환: (비율(%), 대상 건수, 설정 건수)"""
    d = jr[jr["결정"].isin(["매수", "추가 매수"])] if jr is not None and len(jr) else pd.DataFrame()
    if start is not None and len(d):
        d = d[d["날짜"] >= pd.Timestamp(start)]
    if d.empty:
        return np.nan, 0, 0
    ok = int((pd.to_numeric(d["무효가격"], errors="coerce") > 0).sum())
    return ok / len(d) * 100, int(len(d)), ok


def _busdays(a, b):
    return int(np.busday_count(pd.Timestamp(a).date(), pd.Timestamp(b).date()))


def plan_events(plans, closes, today=None, grace=3):
    """각 계획의 트리거(무효가격 이탈, 1차 목표 도달, 기간 만료)와 계획대로 처리했는지.
    결과: 준수(트리거 후 처리 기한 안에 처리), 미준수(기한이 지나도 처리 안 함), 기한 전, 기록 없음(예전 계획이라 처리일 기록이 없음)"""
    t = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    rows = []

    def add(p, kind, date, handled):
        if handled is None or pd.isna(handled):
            res = "기한 전" if _busdays(date, t) <= grace else "미준수"
        else:
            res = "준수" if _busdays(date, handled) <= grace else "미준수"
        rows.append({"계획": p["id"], "종목": p.get("종목명") or p.get("종목코드"), "유형": kind, "발생일": pd.Timestamp(date), "처리일": handled, "결과": res})
    for p in plans.to_dict("records"):
        s = closes.get(str(p["종목코드"]).zfill(6))
        closed = p.get("상태") == "종료"
        if s is not None and len(s):
            seg = s[s.index >= p["생성일"]]
            inv, t1 = p.get("무효가격"), p.get("목표1")
            if inv == inv and inv and len(seg[seg <= inv]):
                d = seg[seg <= inv].index[0]
                h = p.get("종료일")
                if (h is None or pd.isna(h)) and closed:
                    rows.append({"계획": p["id"], "종목": p.get("종목명"), "유형": "무효가격 이탈", "발생일": d, "처리일": pd.NaT, "결과": "기록 없음"})
                else:
                    add(p, "무효가격 이탈", d, h)
            if t1 == t1 and t1 and len(seg[seg >= t1]):
                d = seg[seg >= t1].index[0]
                h = p.get("완료1일")
                if (h is None or pd.isna(h)) and (p.get("완료1") == "Y" or closed):
                    rows.append({"계획": p["id"], "종목": p.get("종목명"), "유형": "1차 목표 도달", "발생일": d, "처리일": pd.NaT, "결과": "기록 없음"})
                else:
                    add(p, "1차 목표 도달", d, h)
        for m in re.finditer(r"\[연장 (\d{4}-\d{2}-\d{2})\]", str(p.get("메모") or "")):        # 연장은 사유를 적고 한 것이므로 '처리'로 본다(원래 만료일은 연장하면 덮어써져서 메모로 확인)
            rows.append({"계획": p["id"], "종목": p.get("종목명"), "유형": "기간 만료", "발생일": pd.Timestamp(m.group(1)), "처리일": pd.Timestamp(m.group(1)), "결과": "준수"})
        exp = p.get("만료일")
        if exp is not None and not pd.isna(exp) and pd.Timestamp(exp) <= t:
            h = p.get("종료일")
            if (h is None or pd.isna(h)) and closed:
                rows.append({"계획": p["id"], "종목": p.get("종목명"), "유형": "기간 만료", "발생일": pd.Timestamp(exp), "처리일": pd.NaT, "결과": "기록 없음"})
            else:
                add(p, "기간 만료", exp, h)
    return pd.DataFrame(rows, columns=["계획", "종목", "유형", "발생일", "처리일", "결과"])


def compliance_summary(ev):
    """유형별·전체 준수율. 기한 전·기록 없음은 분모에서 뺀다."""
    out = []
    if ev is None or ev.empty:
        return pd.DataFrame(columns=["유형", "평가 대상", "준수", "준수율(%)", "기한 전", "기록 없음"])
    for k, g in [("전체", ev)] + [(t, ev[ev["유형"] == t]) for t in ev["유형"].unique()]:
        ok, bad = int((g["결과"] == "준수").sum()), int((g["결과"] == "미준수").sum())
        out.append({"유형": k, "평가 대상": ok + bad, "준수": ok, "준수율(%)": (ok / (ok + bad) * 100) if ok + bad else np.nan,
                    "기한 전": int((g["결과"] == "기한 전").sum()), "기록 없음": int((g["결과"] == "기록 없음").sum())})
    return pd.DataFrame(out)


def reason_compare(ev, h=60):
    """판단 기록의 이유 묶음별 결과(journal.evaluate 결과 사용): 건수, 맞은 비율, 평균 결정 손익, 평균 코스피 대비."""
    col, acol = f"v{h}", f"a{h}"
    if ev is None or ev.empty or col not in ev.columns:
        return pd.DataFrame(columns=["묶음", "건수", "맞은 비율(%)", "평균 결정 손익(%)", "평균 코스피 대비(%p)", "참고"])
    d = ev.dropna(subset=[col]).copy()
    d["묶음"] = d["이유"].map(reason_group)
    rows = []
    for k, g in d.groupby("묶음"):
        v = g[col].astype(float)
        rows.append({"묶음": k, "건수": int(len(v)), "맞은 비율(%)": float((v > 0).mean() * 100), "평균 결정 손익(%)": float(v.mean()),
                     "평균 코스피 대비(%p)": float(g[acol].mean()) if acol in g.columns and g[acol].notna().any() else np.nan, "참고": "표본 적음(참고만)" if len(v) < 20 else ""})
    return pd.DataFrame(rows)
