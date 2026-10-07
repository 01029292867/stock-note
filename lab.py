"""가설 실험실: 흔히 쓰는 기술 지표를 가설로 놓고, 과거 가격에서 실제로 효과가 있었는지 검증한다.
- 종목 × 기간의 이벤트를 모아서(겹치지 않게 간격을 두고) 시장 평균을 뺀 초과수익으로 비교한다.
- 시험한 가설 수만큼 기준을 엄격하게 하고(다중검정 보정), 앞·뒤 기간의 일관성을 본다.
- 과거에 효과가 있었다는 것이 앞으로도 있다는 보장은 아니다. 상장폐지 종목이 빠진 자료는 결과를 좋게 보이게 한다."""
VERSION = 3  # 3: 수급 요인 검증(수급 기록 사용) / 2: 요인(분위) 검증

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
    closes, vols, sigs = {}, {}, {k: {} for k in CATALOG}
    for code, df in prices.items():
        if df is None or len(df) < min_days:
            continue
        ind = indicators(df)
        closes[code] = ind["c"]
        vols[code] = (df["Volume"].astype(float) * df["Close"].astype(float)) if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
        for k, (_, fn) in CATALOG.items():
            sigs[k][code] = fn(ind).fillna(False)
    if not closes:
        return None
    C = pd.DataFrame(closes).sort_index()
    S = {k: pd.DataFrame(v).reindex(C.index).fillna(False).astype(bool) for k, v in sigs.items()}
    fwd = C.shift(-h) / C - 1
    mn = C.shift(-1).rolling(h).min().shift(-(h - 1))           # t+1 ~ t+h 구간의 최저 종가
    V = pd.DataFrame(vols).reindex(C.index)
    return {"C": C, "V": V, "S": S, "fwd": fwd, "ex": fwd.sub(fwd.mean(axis=1), axis=0), "mae": mn / C - 1, "h": h}


def _block_boot(sums, cnts, L, B, rng):
    """연속한 L개 구간을 통째로 다시 뽑는 블록 부트스트랩. 보유 기간이 겹치는 인접 표본의 상관을 반영해서 신뢰구간이 좁아지지 않게 한다."""
    n = len(sums)
    L = max(1, min(int(L), n))
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(B, -1)[:, :n]
    return sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)


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
    boot = _block_boot(sums, cnts, int(np.ceil(P["h"] / 21)) + 1, B, rng)   # 월 단위 블록으로 다시 뽑는다(같은 달의 종목은 함께 움직이고, 보유 기간이 겹치는 인접한 달도 이어져 있다)
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


# ---------------- 요인(분위) 검증 ----------------
# 값이 클수록 '그 성격이 강한' 쪽이다. 결과의 부호가 방향을 알려준다(윗그룹 - 아랫그룹).
FACTORS = {
    "6개월 수익률(모멘텀)": ("최근 6개월(126거래일) 수익률이 높은 종목", lambda C, V: C / C.shift(126) - 1),
    "12개월-1개월 수익률(모멘텀)": ("최근 1개월을 뺀 11개월 수익률이 높은 종목", lambda C, V: C.shift(21) / C.shift(252) - 1),
    "1개월 수익률(단기 반전 가설)": ("최근 1개월 수익률이 높은 종목(내리면 반전 효과)", lambda C, V: C / C.shift(21) - 1),
    "변동성(60일)": ("일간 수익률의 60일 변동이 큰 종목", lambda C, V: C.pct_change().rolling(60).std()),
    "52주 고점 대비 위치": ("종가가 52주 고점에 가까운 종목", lambda C, V: C / C.rolling(252, min_periods=120).max()),
    "거래대금(20일 평균)": ("거래대금이 큰 종목(유동성)", lambda C, V: V.rolling(20).mean()),
}


def factor_study(F, P, step=5, n_q=5, n_tests=1, B=2000, seed=0):
    """매 step일마다 종목을 요인 값 순서로 n_q등분하고 이후 수익률을 비교한다. 윗그룹 - 아랫그룹의 평균과 신뢰구간을 구한다.
    같은 달의 표본끼리는 함께 움직이므로 월 단위로 다시 뽑아서 신뢰구간을 만든다."""
    fwd, dates = P["fwd"], P["C"].index
    rows, ds = [], []
    for i in range(0, len(dates), step):
        f, r = F.iloc[i], fwd.iloc[i]
        ok = f.notna() & r.notna()
        if ok.sum() < 30:
            continue
        rk = f[ok].rank(pct=True, method="first")
        q = np.minimum((rk * n_q).astype(int), n_q - 1) if hasattr(rk, "astype") else rk
        rr = r[ok]
        rows.append([float(rr[q == g].mean()) for g in range(n_q)])
        ds.append(dates[i])
    out = {"n_dates": len(rows)}
    if len(rows) < 30:
        out["grade"] = "표본 부족"
        return out
    Q = np.asarray(rows)
    spread = Q[:, -1] - Q[:, 0]
    rng = np.random.default_rng(seed)
    boot = _block_boot(spread, np.ones(len(spread)), int(np.ceil(2 * P["h"] / step)), B, rng)   # 인접한 표본(보유 기간이 겹침)을 묶어서 뽑는다
    alpha = 0.05 / max(1, n_tests)
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    half = len(spread) // 2
    s1, s2 = float(spread[:half].mean()), float(spread[half:].mean())
    mean_s = float(spread.mean())
    consistent = bool(np.sign(s1) == np.sign(s2) == np.sign(mean_s))
    qm = Q.mean(axis=0) * 100
    mono = float(np.corrcoef(np.arange(n_q), qm)[0, 1])
    out.update({"q": qm, "spread": mean_s * 100, "ci_lo": float(lo * 100), "ci_hi": float(hi * 100), "half1": s1 * 100, "half2": s2 * 100,
                "consistent": consistent, "mono": mono, "alpha": alpha})
    if lo > 0 and consistent:
        out["grade"] = "검증됨(약함): 높을수록 유리"
    elif hi < 0 and consistent:
        out["grade"] = "검증됨(약함): 낮을수록 유리"
    elif mean_s != 0 and consistent:
        out["grade"] = "가설(방향만)"
    else:
        out["grade"] = "효과 구분 안 됨"
    return out


# ---------------- 수급 요인 검증 (구글 시트 '수급기록'이 쌓인 만큼만 검증된다) ----------------
FLOW_FACTORS = {
    "큰손(외국인+기관) 20일 순매수 비중": "외국인·기관이 20일간 순매수한 정도(3주체 거래금액 대비). 높을수록 큰손이 사는 종목",
    "외국인 20일 순매수 비중": "외국인이 20일간 순매수한 정도",
    "기관 20일 순매수 비중": "기관이 20일간 순매수한 정도",
    "개인 20일 순매수 비중(개인 주도 가설)": "개인이 20일간 순매수한 정도. 높을수록 개인 주도 종목(가설: 중장기에는 불리)",
}


def flow_matrices(fl, C, window=20):
    """수급 기록(날짜, 종목코드, 기관(주), 외국인(주), 개인(주), 종가) -> 요인 이름별 (날짜 × 종목) 표. C의 날짜·종목에 맞춘다."""
    d = fl.copy()
    d["날짜"] = pd.to_datetime(d["날짜"], errors="coerce")
    d = d.dropna(subset=["날짜"])
    for col, key in (("기관(주)", "i"), ("외국인(주)", "f"), ("개인(주)", "p")):
        d[key] = pd.to_numeric(d[col], errors="coerce") * pd.to_numeric(d["종가"], errors="coerce")
    piv = {k: d.pivot_table(index="날짜", columns="종목코드", values=k, aggfunc="last") for k in ("i", "f", "p")}
    roll = {}
    for k, v in piv.items():
        roll[k] = v.reindex(index=C.index, columns=C.columns).rolling(window, min_periods=int(window * 0.75)).sum()
    f, i = roll["f"], roll["i"]
    p_ = roll["p"].where(roll["p"].notna(), -(f + i))          # 개인 값이 없으면 -(외국인+기관)으로 추정
    den = f.abs() + i.abs() + p_.abs()
    den = den.where(den > 0)
    return {"큰손(외국인+기관) 20일 순매수 비중": (f + i) / den, "외국인 20일 순매수 비중": f / den,
            "기관 20일 순매수 비중": i / den, "개인 20일 순매수 비중(개인 주도 가설)": p_ / den}


def flow_readiness(fl, C):
    """수급 기록이 검증에 쓸 만한지: (겹치는 종목 수, 기록된 거래일 수, 가장 오래된 날짜, 종목 30개 이상인 날짜 수)"""
    if fl is None or fl.empty:
        return 0, 0, None, 0
    d = fl.copy()
    d["날짜"] = pd.to_datetime(d["날짜"], errors="coerce")
    d = d[d["종목코드"].isin(C.columns)].dropna(subset=["날짜"])
    if d.empty:
        return 0, 0, None, 0
    per_day = d.groupby("날짜")["종목코드"].nunique()
    return int(d["종목코드"].nunique()), int(d["날짜"].nunique()), d["날짜"].min().date().isoformat(), int((per_day >= 30).sum())
