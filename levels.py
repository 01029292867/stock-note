"""가격 흐름에서 지지선·저항선·평소 되돌림 폭을 찾는다(종가 기준 지그재그).
과거에 밀렸다가 반등한 지점을 참고용으로 정리한 것이고, 같은 가격대에서 다시 반등한다는 보장은 없다."""
VERSION = 1

import numpy as np

CLUSTER = 0.025   # 이 범위 안의 저점·고점은 같은 가격대로 묶는다
BREAK = 0.02      # 지지선 아래로 이만큼 종가가 내려가야 이탈로 본다


def auto_threshold(closes, tag="중장기"):
    """종목의 평소 변동성에 맞춘 반전 기준(고점에서 이만큼 내려오면 한 번 밀렸다고 본다)."""
    c = np.asarray(closes, float)
    if len(c) < 30:
        return 0.08
    r = np.diff(c) / c[:-1]
    thr = float(np.clip(np.std(r) * np.sqrt(20) * 1.1, 0.06, 0.20))
    return float(max(0.04, thr * (0.6 if tag == "스윙" else 1.0)))


def zigzag(c, thr):
    """종가 지그재그. 반환: [{'i','type'('H'/'L'),'price','provisional'}] (마지막 하나는 진행 중인 극값)."""
    n = len(c)
    if n < 3:
        return []
    piv, trend, ext, hi, lo = [], 0, 0, 0, 0
    for i in range(1, n):
        if trend == 0:
            if c[i] > c[hi]:
                hi = i
            if c[i] < c[lo]:
                lo = i
            if hi > lo and c[hi] >= c[lo] * (1 + thr):
                piv.append({"i": lo, "type": "L", "price": float(c[lo]), "provisional": False}); trend, ext = 1, hi
            elif lo > hi and c[lo] <= c[hi] * (1 - thr):
                piv.append({"i": hi, "type": "H", "price": float(c[hi]), "provisional": False}); trend, ext = -1, lo
        elif trend == 1:
            if c[i] > c[ext]:
                ext = i
            elif c[i] <= c[ext] * (1 - thr):
                piv.append({"i": ext, "type": "H", "price": float(c[ext]), "provisional": False}); trend, ext = -1, i
        else:
            if c[i] < c[ext]:
                ext = i
            elif c[i] >= c[ext] * (1 + thr):
                piv.append({"i": ext, "type": "L", "price": float(c[ext]), "provisional": False}); trend, ext = 1, i
    if trend != 0:
        piv.append({"i": ext, "type": "H" if trend == 1 else "L", "price": float(c[ext]), "provisional": True})
    return piv


def pullbacks(piv, dates):
    """고점 -> 저점 -> 반등 구간 목록(오래된 것부터)."""
    out = []
    for k in range(len(piv) - 1):
        a, b = piv[k], piv[k + 1]
        if a["type"] == "H" and b["type"] == "L":
            nxt = piv[k + 2] if k + 2 < len(piv) else None
            reb = (nxt["price"] / b["price"] - 1) * 100 if nxt else np.nan
            out.append({"peak_date": dates[a["i"]], "peak": a["price"], "trough_date": dates[b["i"]], "trough": b["price"],
                        "depth": (b["price"] / a["price"] - 1) * 100, "rebound": reb,
                        "rebound_days": (nxt["i"] - b["i"]) if nxt else None, "ongoing": b["provisional"] or (nxt is not None and nxt["provisional"])})
    return out


def _cluster(levels):
    """[(가격, 날짜, 반등%)] -> 가까운 가격대끼리 묶은 목록."""
    levels = sorted(levels, key=lambda x: x[0])
    groups, cur = [], []
    for lv in levels:
        if cur and lv[0] <= cur[0][0] * (1 + CLUSTER):
            cur.append(lv)
        else:
            if cur:
                groups.append(cur)
            cur = [lv]
    if cur:
        groups.append(cur)
    return [{"price": float(np.median([x[0] for x in g])), "touches": len(g), "last_date": max(x[1] for x in g),
             "rebound": float(np.nanmax([x[2] for x in g])) if any(x[2] == x[2] for x in g) else np.nan} for g in groups]


def analyze(closes, dates, price, tag="중장기", win=10):
    c = np.asarray(closes, float)
    if len(c) < 40:
        return None
    thr = auto_threshold(c, tag)
    piv = zigzag(c, thr)
    if len(piv) < 3:
        return None
    pbs = pullbacks(piv, dates)
    done = [p for p in pbs if not p["ongoing"]]
    depths = [p["depth"] for p in done]
    # 저점(L)들의 가격대 묶음: 이후 반등이 있었던 저점만
    lows = []
    for k, p in enumerate(piv):
        if p["type"] == "L":
            nxt = piv[k + 1] if k + 1 < len(piv) else None
            reb = (nxt["price"] / p["price"] - 1) * 100 if nxt else np.nan
            lows.append((p["price"], dates[p["i"]], reb, p["i"]))
    sup, broken = [], []
    n = len(c)
    for g in _cluster([(x[0], x[1], x[2]) for x in lows]):
        lvl = g["price"]
        first_i = min(x[3] for x in lows if abs(x[0] / lvl - 1) <= CLUSTER)
        after = np.where(c[first_i + 1:] < lvl * (1 - BREAK))[0]
        bi = int(first_i + 1 + after[0]) if len(after) else None
        if bi is None and lvl < price:
            sup.append(dict(g, dist=(lvl / price - 1) * 100))
        elif bi is not None and price < lvl * (1 - BREAK / 2):
            broken.append(dict(g, break_date=dates[bi], days=n - 1 - bi, dist=(lvl / price - 1) * 100))
    highs = [(p["price"], dates[p["i"]], np.nan) for p in piv if p["type"] == "H"]
    res = [dict(g, dist=(g["price"] / price - 1) * 100) for g in _cluster(highs) if g["price"] > price * (1 + BREAK / 2)]
    sup = sorted(sup, key=lambda x: -x["price"])
    res = sorted(res, key=lambda x: x["price"])
    broken = sorted(broken, key=lambda x: x["days"])
    # 마지막 고점 이후 현재까지의 하락
    peaks = [p for p in piv if p["type"] == "H"]
    last_peak = peaks[-1] if peaks else None
    cur_dd = (price / last_peak["price"] - 1) * 100 if last_peak else np.nan
    return {"thr": thr, "pivots": piv, "pullbacks": pbs, "depth_med": float(np.median(depths)) if depths else np.nan,
            "depth_p75": float(np.percentile(depths, 25)) if depths else np.nan, "n_pull": len(depths),
            "supports": sup[:3], "resistances": res[:2], "broken": broken, "recent_break": [b for b in broken if b["days"] <= win],
            "last_peak": last_peak["price"] if last_peak else None, "last_peak_date": dates[last_peak["i"]] if last_peak else None, "cur_dd": cur_dd}
