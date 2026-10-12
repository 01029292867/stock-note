"""파동(박스권) 분석: 오르내림을 반복하는 종목을 숫자로 찾고, 진입·매도 구간과 무효가격(박스권 이탈)을 계산한다.
과거에 이 규칙을 썼다면 어땠는지(백테스트)와, 파동형으로 보이는 종목에서 정말 더 잘 통했는지(전 종목 검증)도 같이 본다.
박스권은 언제든 깨질 수 있고(추세 전환), 비용(약 0.3%)보다 파동 폭이 충분히 커야 남는다. 과거 결과가 미래를 보장하지 않는다."""
VERSION = 1

import numpy as np
import pandas as pd

import levels

COST = 0.003


def _cv(x):
    x = np.asarray(x, float)
    return float(np.std(x) / abs(np.mean(x))) if len(x) >= 2 and np.mean(x) != 0 else np.nan


def profile(close, window=250, band_win=120, lo_q=0.2, hi_q=0.8, cost=COST):
    """종목의 파동 성격. 반환 dict 또는 None(자료가 짧을 때)."""
    c = pd.Series(close).dropna().astype(float)
    if len(c) < 130:
        return None
    w = c.iloc[-window:] if len(c) >= window else c
    y = np.log(w.values)
    x = np.arange(len(y))
    slope, icpt = np.polyfit(x, y, 1)
    fit = slope * x + icpt
    r2 = float(1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if ((y - y.mean()) ** 2).sum() > 0 else 0.0
    trend = float((np.exp(slope * 252) - 1) * 100)
    vol = float(np.std(np.diff(y)))
    thr = float(np.clip(vol * np.sqrt(20) * 0.9, 0.05, 0.15))
    piv = levels.zigzag(w.values, thr)
    ups, downs, gaps = [], [], []
    for a, b in zip(piv, piv[1:]):
        if b["provisional"]:
            continue
        chg = (b["price"] / a["price"] - 1) * 100
        gaps.append(b["i"] - a["i"])
        (ups if a["type"] == "L" else downs).append(chg)
    per_year = (len(ups) + len(downs)) / (len(w) / 252)
    amp = float(np.median(ups)) if ups else np.nan
    b = c.iloc[-band_win:]
    L, U, lo, hi = float(b.quantile(lo_q)), float(b.quantile(hi_q)), float(b.min()), float(b.max())
    price = float(c.iloc[-1])
    pos = float((price - lo) / (hi - lo) * 100) if hi > lo else np.nan
    gain = (U / L - 1 - cost) * 100
    s_trend = 1 - min(abs(trend) / 30, 1)
    s_r2 = 1 - min(r2 / 0.5, 1)
    s_freq = min(per_year / 6, 1)
    s_amp = min(max(amp - 3, 0) / 9, 1) if amp == amp else 0.0
    cv = _cv(ups)
    s_reg = 1 - min(cv, 1.0) if cv == cv else 0.0
    score = float(100 * (0.30 * s_trend + 0.15 * s_r2 + 0.20 * s_freq + 0.25 * s_amp + 0.10 * s_reg))
    parts = [("추세가 약한가", f"연 {trend:+.0f}%", s_trend, 0.30, "연 ±30%를 넘으면 0점"),
             ("한 방향으로 흐르지 않는가", f"추세 설명력 {r2 * 100:.0f}%", s_r2, 0.15, "설명력 50% 이상이면 0점"),
             ("오르내림이 자주 있는가", f"연 {per_year:.1f}회", s_freq, 0.20, "연 6회 이상이면 만점"),
             ("한 번 움직이는 폭이 큰가", "-" if amp != amp else f"+{amp:.1f}%", s_amp, 0.25, "3%면 0점, 12% 이상이면 만점"),
             ("폭이 규칙적인가", "-" if cv != cv else f"변동계수 {cv:.2f}", s_reg, 0.10, "폭이 들쭉날쭉하면 낮아요")]
    state = "진입 구간" if pos == pos and pos <= 25 else "매도 구간" if pos == pos and pos >= 75 else "중간"
    return {"score": score, "trend": trend, "r2": r2, "thr": thr * 100, "n_up": len(ups), "n_down": len(downs), "per_year": float(per_year),
            "amp_up": amp, "amp_down": float(np.median(downs)) if downs else np.nan, "period": float(np.median(gaps)) if gaps else np.nan,
            "L": L, "U": U, "lo": lo, "hi": hi, "pos": pos, "gain": float(gain), "price": price, "state": state,
            "invalid": float(lo * 0.97), "buy_hi": L, "sell_lo": U, "parts": parts}


def backtest(close, band_win=120, lo_q=0.2, hi_q=0.8, stop_gap=0.03, max_hold=60, min_gain=0.06, cost=COST, start=None):
    """최근 band_win일의 하단(lo_q)·상단(hi_q) 가격을 구간으로 보고, 하단 이하면 다음 날 종가에 사서 상단에서 팔고,
    하단보다 stop_gap 더 내려가면 손절, max_hold일이 지나면 정리한다. 미래 값을 쓰지 않는다. 반환: 거래 DataFrame"""
    c = pd.Series(close).dropna().astype(float)
    v, idx = c.values, c.index
    n = len(v)
    trades, t, i0 = [], band_win, (band_win if start is None else max(band_win, int(start)))
    t = i0
    while t < n - 2:
        b = v[t - band_win:t]
        L, U = np.quantile(b, lo_q), np.quantile(b, hi_q)
        if v[t] <= L and (U / L - 1) >= min_gain:
            e = t + 1                                    # 신호가 난 다음 날 종가에 산다
            entry, stop, target = v[e], L * (1 - stop_gap), U
            k, reason = e, "기간 종료"
            for k in range(e + 1, min(e + max_hold, n - 1) + 1):
                if v[k] >= target:
                    reason = "목표 도달"
                    break
                if v[k] <= stop:
                    reason = "박스권 이탈 손절"
                    break
            else:
                k = min(e + max_hold, n - 1)
                reason = "보유 기간 종료"
            ret = v[k] / entry - 1 - cost
            trades.append({"진입일": idx[e], "청산일": idx[k], "진입가": entry, "청산가": v[k], "수익률(%)": ret * 100, "보유일": k - e, "사유": reason})
            t = k + 1
        else:
            t += 1
    return pd.DataFrame(trades, columns=["진입일", "청산일", "진입가", "청산가", "수익률(%)", "보유일", "사유"])


def stats(trades, close=None):
    """거래 결과 요약: 거래 수, 승률, 평균 이익·손실, 손익비, 건당 기대값, 누적 수익, 보유 기간 비중."""
    if trades is None or trades.empty:
        return {"n": 0}
    r = trades["수익률(%)"].values / 100
    w, l = r[r > 0], r[r < 0]
    out = {"n": int(len(r)), "win": float((r > 0).mean() * 100), "avg_win": float(w.mean() * 100) if len(w) else np.nan,
           "avg_loss": float(l.mean() * 100) if len(l) else np.nan, "exp": float(r.mean() * 100), "cum": float((np.prod(1 + r) - 1) * 100),
           "days": float(trades["보유일"].mean()), "stops": int((trades["사유"] == "박스권 이탈 손절").sum())}
    out["payoff"] = out["avg_win"] / abs(out["avg_loss"]) if (out["avg_win"] == out["avg_win"] and out["avg_loss"] == out["avg_loss"] and out["avg_loss"]) else np.nan
    if close is not None and len(close):
        c = pd.Series(close).dropna()
        first = trades["진입일"].iloc[0]
        base = c[c.index >= first]
        out["bh"] = float((base.iloc[-1] / base.iloc[0] - 1) * 100) if len(base) > 1 else np.nan
        out["exposure"] = float(trades["보유일"].sum() / max(len(base), 1) * 100)
    return out


def wave_study(prices, formation=250, n_boot=1000, seed=0, **bt):
    """전 종목 검증: 처음 formation일로 파동 점수를 매기고(미래 값 없이), 그 뒤 기간에 같은 규칙을 적용해서
    파동 점수 상위 그룹이 하위 그룹보다 실제로 더 잘 통했는지 본다. 반환: (그룹별 표, 요약 dict)"""
    rows = []
    for code, df in prices.items():
        c = pd.Series(df["Close"]).dropna() if df is not None and len(df) else None
        if c is None or len(c) < formation + 250:
            continue
        pf = profile(c.iloc[:formation], window=formation)
        if pf is None:
            continue
        tr = backtest(c, start=formation, **bt)
        rows.append({"code": code, "score": pf["score"], "trades": tr, "bh": float((c.iloc[-1] / c.iloc[formation] - 1) * 100)})
    if len(rows) < 20:
        return None, {"n_stocks": len(rows), "msg": "종목이 너무 적어서(20개 미만) 비교할 수 없어요."}
    sc = np.array([r["score"] for r in rows])
    q = np.quantile(sc, [0.2, 0.4, 0.6, 0.8])
    grp = np.digitize(sc, q)                                      # 0(하위)~4(상위)
    names = ["1(파동 약함)", "2", "3", "4", "5(파동 강함)"]

    def pooled(sel):
        allr = np.concatenate([rows[i]["trades"]["수익률(%)"].values for i in sel if len(rows[i]["trades"])] or [np.array([])])
        return allr

    out, per_stock_exp = [], {}
    for g in range(5):
        sel = [i for i in range(len(rows)) if grp[i] == g]
        r = pooled(sel)
        st_ = {"그룹": names[g], "종목수": len(sel), "거래 수": int(len(r))}
        if len(r):
            w, l = r[r > 0], r[r < 0]
            st_.update({"승률(%)": float((r > 0).mean() * 100), "평균 이익(%)": float(w.mean()) if len(w) else np.nan, "평균 손실(%)": float(l.mean()) if len(l) else np.nan,
                        "건당 기대값(%)": float(r.mean()), "박스권 이탈 손절 비율(%)": float(np.mean(np.concatenate([rows[i]["trades"]["사유"].values for i in sel if len(rows[i]["trades"])]) == "박스권 이탈 손절") * 100)})
        st_["같은 기간 보유 평균(%)"] = float(np.mean([rows[i]["bh"] for i in sel])) if sel else np.nan
        out.append(st_)
        per_stock_exp[g] = [(rows[i]["trades"]["수익률(%)"].mean() if len(rows[i]["trades"]) else np.nan) for i in sel]
    rng = np.random.default_rng(seed)
    top, bot = np.array([x for x in per_stock_exp[4] if x == x]), np.array([x for x in per_stock_exp[0] if x == x])
    summ = {"n_stocks": len(rows)}
    if len(top) >= 5 and len(bot) >= 5:
        diffs = [rng.choice(top, len(top)).mean() - rng.choice(bot, len(bot)).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        summ.update({"spread": float(top.mean() - bot.mean()), "ci": (float(lo), float(hi)), "distinct": bool(lo > 0 or hi < 0)})
    allr = pooled(range(len(rows)))
    if len(allr):
        summ["all_exp"], summ["all_win"] = float(allr.mean()), float((allr > 0).mean() * 100)
    return pd.DataFrame(out), summ
