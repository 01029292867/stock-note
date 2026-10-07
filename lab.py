"""가설 실험실: 흔히 쓰는 기술 지표를 가설로 놓고, 과거 가격에서 실제로 효과가 있었는지 검증한다.
- 종목 × 기간의 이벤트를 모아서(겹치지 않게 간격을 두고) 시장 평균을 뺀 초과수익으로 비교한다.
- 시험한 가설 수만큼 기준을 엄격하게 하고(다중검정 보정), 앞·뒤 기간의 일관성을 본다.
- 과거에 효과가 있었다는 것이 앞으로도 있다는 보장은 아니다. 상장폐지 종목이 빠진 자료는 결과를 좋게 보이게 한다."""
VERSION = 1

import numpy as np
import pandas as pd


def indicators(df):
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(np.nan, index=c.index)
    ma20, ma60, ma120 = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(120).mean()
    sd20 = c.rolling(20).std()
    d = c.diff()
    gain, loss = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.where(loss != 0, 100.0).where(loss.notna())
    return pd.DataFrame({
        "c": c, "ma20": ma20, "ma60": ma60, "ma120": ma120, "bb_up": ma20 + 2 * sd20, "bb_lo": ma20 - 2 * sd20,
        "gap20": c / ma20 * 100, "rsi": rsi, "hi": c.rolling(252, min_periods=120).max(), "lo": c.rolling(252, min_periods=120).min(),
        "volx": v / v.rolling(20).mean()})


def _recent(b, n):
    return b.astype(float).rolling(n, min_periods=1).max() > 0


CATALOG = {
    "20일선 위": ("종가가 20일 이동평균선 위", lambda i: i.c > i.ma20),
    "20일선 아래": ("종가가 20일 이동평균선 아래", lambda i: i.c < i.ma20),
    "60일선 위": ("종가가 60일 이동평균선 위", lambda i: i.c > i.ma60),
    "60일선 아래": ("종가가 60일 이동평균선 아래", lambda i: i.c < i.ma60),
    "정배열": ("20일선 > 60일선 > 120일선이고 종가가 20일선 위", lambda i: (i.ma20 > i.ma60) & (i.ma60 > i.ma120) & (i.c > i.ma20)),
    "골든크로스(최근 5일)": ("20일선이 60일선을 위로 뚫음", lambda i: _recent((i.ma20 > i.ma60) & (i.ma20.shift(1) <= i.ma60.shift(1)), 5)),
    "데드크로스(최근 5일)": ("20일선이 60일선을 아래로 뚫음", lambda i: _recent((i.ma20 < i.ma60) & (i.ma20.shift(1) >= i.ma60.shift(1)), 5)),
    "볼린저 하단 이탈 후 복귀": ("밴드 하단 아래로 내려갔다가 3일 안에 밴드 안으로 복귀", lambda i: _recent((i.c < i.bb_lo).shift(1).fillna(False), 3) & (i.c >= i.bb_lo)),
    "볼린저 상단 돌파": ("종가가 밴드 상단 위", lambda i: i.c > i.bb_up),
    "이격도 과열(≥110)": ("종가가 20일선보다 10% 이상 위", lambda i: i.gap20 >= 110),
    "이격도 침체(≤92)": ("종가가 20일선보다 8% 이상 아래", lambda i: i.gap20 <= 92),
    "RSI 과매도(≤30)": ("RSI 30 이하", lambda i: i.rsi <= 30),
    "RSI 과열(≥70)": ("RSI 70 이상", lambda i: i.rsi >= 70),
    "52주 신고가 근접(≥95%)": ("종가가 52주 고점의 95% 이상", lambda i: i.c >= 0.95 * i.hi),
    "52주 저점권(저점+20% 이내)": ("종가가 52주 저점의 120% 이하", lambda i: i.c <= 1.2 * i.lo),
    "거래량 급증 상승": ("거래량이 20일 평균의 2배 이상이면서 상승", lambda i: (i.volx >= 2) & (i.c > i.c.shift(1))),
}
# 종목 리포트의 신호 이름 -> 위 가설 이름
SIGNAL_TO_HYP = {"20일선 이탈": "20일선 아래", "RSI 과열": "RSI 과열(≥70)", "52주 고점권": "52주 신고가 근접(≥95%)"}


def build(prices, h, min_days=260):
    """prices: {종목코드: DataFrame(Close, Volume)} -> (종가, 신호들, 이후 수익률, 시장 대비 초과수익, 구간 최대 하락)"""
    closes, sigs = {}, {k: {} for k in CATALOG}
    for code, df in prices.items():
        if df is None or len(df) < min_days:
            continue
        ind = indicators(df)
        closes[code] = ind["c"]
        for k, (_, fn) in CATALOG.items():
            sigs[k][code] = fn(ind).fillna(False)
    if not closes:
        return None
    C = pd.DataFrame(closes).sort_index()
    S = {k: pd.DataFrame(v).reindex(C.index).fillna(False).astype(bool) for k, v in sigs.items()}
    fwd = C.shift(-h) / C - 1
    mn = C.shift(-1).rolling(h).min().shift(-(h - 1))           # t+1 ~ t+h 구간의 최저 종가
    return {"C": C, "S": S, "fwd": fwd, "ex": fwd.sub(fwd.mean(axis=1), axis=0), "mae": mn / C - 1, "h": h}


def _thin(valid, h):
    idx, sel, last = np.flatnonzero(valid), [], -10 ** 9
    for i in idx:
        if i - last >= h:
            sel.append(i)
            last = i
    return np.asarray(sel, dtype=int)


def _collect(mask, P):
    fw, ex, mae, dates = [], [], [], []
    M, F, E, A = mask.values, P["fwd"].values, P["ex"].values, P["mae"].values
    ok = ~np.isnan(F)
    for j in range(M.shape[1]):
        sel = _thin(M[:, j] & ok[:, j], P["h"])
        if len(sel):
            fw.append(F[sel, j]); ex.append(E[sel, j]); mae.append(A[sel, j]); dates.append(P["C"].index[sel])
    if not fw:
        return np.array([]), np.array([]), np.array([]), pd.DatetimeIndex([])
    return np.concatenate(fw), np.concatenate(ex), np.concatenate(mae), pd.DatetimeIndex(np.concatenate([d.values for d in dates]))


def event_study(mask, P, cost=0.003, n_tests=1, B=2000, seed=0):
    """mask: 신호 불리언 표(날짜×종목). 반환: 통계 dict."""
    fw, ex, mae, dt_ = _collect(mask, P)
    bw, bex, _, _ = _collect(pd.DataFrame(True, index=mask.index, columns=mask.columns), P)
    out = {"n": int(len(fw)), "n_base": int(len(bw))}
    if len(fw) < 30:
        out["grade"] = "표본 부족"
        return out
    net, bnet = fw - cost, bw - cost
    p1, p0 = float((net > 0).mean()), float((bnet > 0).mean())
    se = np.sqrt(p1 * (1 - p1) / len(net) + p0 * (1 - p0) / len(bnet))
    months = np.asarray(dt_.to_period("M").astype(str))
    um, inv = np.unique(months, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=ex), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(um), size=(B, len(um)))
    boot = sums[draw].sum(axis=1) / cnts[draw].sum(axis=1)       # 월 단위로 다시 뽑아서(같은 달의 종목들은 함께 움직이므로) 평균 초과수익의 분포를 만든다
    alpha = 0.05 / max(1, n_tests)
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    mid = np.median(dt_.values.astype("int64")) if len(dt_) else 0
    first = dt_.values.astype("int64") <= mid
    e1, e2 = (float(ex[first].mean()) if first.any() else np.nan), (float(ex[~first].mean()) if (~first).any() else np.nan)
    mean_ex = float(ex.mean())
    consistent = bool(np.sign(e1) == np.sign(e2) and np.sign(e1) == np.sign(mean_ex))
    out.update({"win": p1 * 100, "win_base": p0 * 100, "win_diff": (p1 - p0) * 100, "mde": 2.8 * se * 100,
                "mean_net": float(net.mean() * 100), "mean_excess": mean_ex * 100, "ci_lo": float(lo * 100), "ci_hi": float(hi * 100),
                "mean_mae": float(np.nanmean(mae) * 100), "half1": e1 * 100, "half2": e2 * 100, "consistent": consistent, "alpha": alpha,
                "avg_win": float(net[net > 0].mean() * 100) if (net > 0).any() else np.nan, "avg_loss": float(net[net < 0].mean() * 100) if (net < 0).any() else np.nan})
    out["grade"] = grade(out)
    return out


def grade(o):
    if o.get("n", 0) < 200:
        return "표본 부족"
    if o["ci_lo"] > 0 and o["consistent"]:
        return "검증됨(약함)"
    if o["ci_hi"] < 0 and o["consistent"]:
        return "불리함 확인(피하는 근거)"
    if o["mean_excess"] > 0 and o["consistent"]:
        return "가설(방향만)"
    return "효과 구분 안 됨"


def combine(P, names):
    m = None
    for n in names:
        m = P["S"][n] if m is None else (m & P["S"][n])
    return m
