"""가격 기반 신호 계산 (시세만으로 판단 가능한 규칙)."""
VERSION = 2  # 2: 매수일 이후 고점(indicators의 since), 수익 보호선(protect), 수급 신호

import numpy as np
import pandas as pd

DEFAULT_RULES = {
    "중장기": {"점검선": -20.0, "목표": 30.0, "고점권": 90.0, "보호활성": 30.0, "보호폭": 20.0},
    "스윙": {"손절선": -7.0, "목표": 15.0, "RSI과열": 70.0, "눌림_하단": 35.0, "눌림_상단": 55.0, "보호활성": 10.0, "보호폭": 8.0},
}


def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    gain = d.clip(lower=0).tail(n).sum()
    loss = (-d.clip(upper=0)).tail(n).sum()
    if loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def indicators(df: pd.DataFrame, since=None) -> dict | None:
    """df: 날짜 인덱스, 'Close' 컬럼. 최소 61행 필요. since(매수일)가 있으면 그 이후의 고점을 쓴다."""
    if df is None or df.empty or "Close" not in df or len(df) < 61:
        return None
    c = df["Close"].astype(float).dropna()
    if len(c) < 61:
        return None
    last = c.iloc[-1]
    yr = c.tail(250)
    hi, lo = yr.max(), yr.min()
    seg, basis = yr, "최근 1년"
    ts = pd.to_datetime(since, errors="coerce") if since else pd.NaT
    if pd.notna(ts):
        s2 = c[c.index >= ts]
        if len(s2):
            seg, basis = s2, "매수일 이후"
    return {
        "peak": float(seg.max()), "peak_date": seg.idxmax().date().isoformat(), "peak_basis": basis,
        "price": float(last),
        "prev": float(c.iloc[-2]),
        "ma20": float(c.tail(20).mean()),
        "ma60": float(c.tail(60).mean()),
        "rsi": rsi(c),
        "hi": float(hi),
        "lo": float(lo),
        "pos": float((last - lo) / (hi - lo) * 100) if hi > lo else 50.0,
        "m1": float((last / c.iloc[-21] - 1) * 100) if len(c) > 21 else np.nan,
        "m3": float((last / c.iloc[-61] - 1) * 100),
        "day": float((last / c.iloc[-2] - 1) * 100),
    }


def protect(ind: dict, tag: str, avg, rules: dict) -> dict | None:
    """수익 보호선: 고점 수익률이 활성 기준 이상이 된 뒤, 고점 대비 보호폭만큼 내려가면 도달."""
    if not ind or not avg or avg <= 0 or not ind.get("peak"):
        return None
    R = rules[tag]
    act, w = R.get("보호활성"), R.get("보호폭")
    if act is None or w is None:
        return None
    peak_ret = (ind["peak"] / avg - 1) * 100
    active = peak_ret >= act
    line = ind["peak"] * (1 - w / 100)
    return {"active": active, "line": line, "dd": (ind["price"] / ind["peak"] - 1) * 100, "peak": ind["peak"],
            "peak_date": ind.get("peak_date"), "peak_ret": peak_ret, "width": w, "act": act,
            "hit": bool(active and ind["price"] <= line), "basis": ind.get("peak_basis")}


def flow_of(ind: dict) -> tuple[str, str]:
    p, a, b = ind["price"], ind["ma20"], ind["ma60"]
    up20, up60, order = p > a, p > b, a > b
    if up20 and up60 and order:
        return "상승 흐름", "현재가가 20일선·60일선 위에 있고, 단기선이 장기선 위에 있어요."
    if not up20 and not up60 and not order:
        return "하락 흐름", "현재가가 20일선·60일선 아래에 있고, 단기선도 장기선 아래예요."
    if up60 and not up20:
        return "조정 중", "60일선 위지만 20일선은 아래예요. 눌림인지 추세 꺾임인지 살펴볼 때예요."
    if up20 and not up60:
        return "반등 시도", "20일선 위로 올라왔지만 60일선은 아직 아래예요."
    return "방향 불분명", "이동평균선이 얽혀 있어 뚜렷한 방향이 없어요."


def signals(ind: dict, tag: str, avg: float | None, rules: dict, flow: dict | None = None) -> list[dict]:
    out = []
    R = rules[tag]
    ret = (ind["price"] / avg - 1) * 100 if avg else None
    if ret is not None:
        key = "손절선" if tag == "스윙" else "점검선"
        if ret <= R[key]:
            out.append({"kind": "주의", "title": f"{key} 도달", "detail": f"수익률 {ret:.1f}% (기준 {R[key]:.0f}%)"})
        if ret >= R["목표"]:
            out.append({"kind": "알림", "title": "목표수익률 도달", "detail": f"수익률 +{ret:.1f}% (기준 +{R['목표']:.0f}%)"})
    pr = protect(ind, tag, avg, rules)
    if pr and pr["hit"]:
        out.append({"kind": "주의", "title": "수익 보호선 도달",
                    "detail": f"고점 {pr['peak']:,.0f}원({pr['peak_date']}) 대비 {pr['dd']:.1f}% · 보호폭 {pr['width']:.0f}% · 고점 기준 수익률 +{pr['peak_ret']:.0f}%"})
    if tag == "중장기":
        if ind["pos"] >= R["고점권"]:
            out.append({"kind": "주의", "title": "52주 고점권", "detail": f"52주 범위 중 {ind['pos']:.0f}% 지점"})
    else:
        if ind["price"] < ind["ma20"]:
            gap = (1 - ind["price"] / ind["ma20"]) * 100
            out.append({"kind": "주의", "title": "20일선 이탈", "detail": f"현재가가 20일선보다 {gap:.1f}% 아래"})
        if ind["rsi"] >= R["RSI과열"]:
            out.append({"kind": "주의", "title": "RSI 과열", "detail": f"RSI {ind['rsi']:.0f} (기준 {R['RSI과열']:.0f} 이상)"})
        if flow and flow.get("f5") is not None and flow["f5"] < 0 and flow["i5"] < 0:
            out.append({"kind": "주의", "title": "외국인·기관 동반 순매도", "detail": f"5일 합계 외국인 {flow['f5']:+,.0f}억 · 기관 {flow['i5']:+,.0f}억"})
        if ind["price"] > ind["ma60"] and R["눌림_하단"] <= ind["rsi"] <= R["눌림_상단"] and ind["price"] > ind["prev"]:
            out.append({"kind": "매수검토", "title": "눌림목 반등 후보", "detail": f"60일선 위 · RSI {ind['rsi']:.0f} · 전일 대비 상승"})
    return out


def verdict(sigs: list[dict]) -> str:
    warn = [s for s in sigs if s["kind"] == "주의"]
    buy = [s for s in sigs if s["kind"] == "매수검토"]
    tgt = [s for s in sigs if s["kind"] == "알림"]
    stop = any("선 도달" in s["title"] for s in warn)
    if stop or len(warn) >= 2:
        return "매도·비중 축소 검토"
    if buy and not warn:
        return "추가매수 검토"
    if tgt:
        return "수익실현 검토"
    if warn:
        return "지켜보기"
    return "보유 유지"
