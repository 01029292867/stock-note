"""종합 판단 카드: 보수적인 상승 여력과 하락 위험, 손익비, 손익분기 승률, 내 평단 기준 시나리오.
예측이 아니라 '내가 가진 근거로 잡은 최악과 최선 사이의 지도'다. 확률은 검증되기 전까지 모른다고 보고,
대신 '이 손익비라면 몇 % 이상 맞혀야 본전인가(필요 승률)'를 보여준다."""
VERSION = 1

import numpy as np

GRADES = [(2.0, "유리"), (1.2, "보통"), (0.7, "다소 불리"), (-1.0, "불리")]


def grade(ratio):
    if ratio is None or ratio != ratio:
        return "근거 부족"
    return next(name for lim, name in GRADES if ratio >= lim)


def build_card(price, avg, qty, tag, ind, lv, cons, flow, safety_counts, rules, total_assets=None):
    """모든 입력은 비어 있을 수 있다(None). 비어 있으면 그 근거는 건너뛰고 '근거 충족도'에 반영한다."""
    ups, downs, warns = [], [], []
    # ---- 상승 후보 (core=True인 것 중 가장 작은 값을 보수적 상승 여력으로 쓴다)
    if cons and cons.get("has") and cons.get("target"):
        full = (cons["target"] / price - 1) * 100
        half = max(full, 0.0) * 0.5
        ups.append({"label": "컨센서스 목표가의 절반만 반영", "pct": half, "price": price * (1 + half / 100), "core": True,
                    "note": f"평균 목표가 {cons['target']:,.0f}원({full:+.0f}%)" + (" — 현재가 이하라 여력 0으로 봐요" if full <= 0 else "")})
    if lv and lv.get("resistances"):
        r = lv["resistances"][0]
        ups.append({"label": "가까운 저항선", "pct": r["dist"], "price": r["price"], "core": True, "note": f"과거 {r['last_date']} 부근에서 막혔던 가격대"})
    if lv and lv.get("last_peak") and lv["last_peak"] > price:
        ups.append({"label": "최근 고점 회복", "pct": (lv["last_peak"] / price - 1) * 100, "price": lv["last_peak"], "core": False, "note": f"{lv['last_peak_date']} 고점"})
    if avg and avg > price:
        ups.append({"label": "내 평단 회복(본전)", "pct": (avg / price - 1) * 100, "price": avg, "core": False, "note": "시장은 내 평단을 신경 쓰지 않아요. 참고용이에요"})
    core = [u["pct"] for u in ups if u["core"] and u["pct"] >= 0]
    up = min(core) if core else None
    up_price = price * (1 + up / 100) if up is not None else None
    # ---- 하락 후보
    if lv and lv.get("supports"):
        s = lv["supports"][0]
        downs.append({"label": "가까운 지지선", "term": "단기", "price": s["price"], "pct": s["dist"], "note": f"과거 {s['touches']}번 반등한 가격대"})
        if len(lv["supports"]) > 1:
            s2 = lv["supports"][1]
            downs.append({"label": "그 아래 지지선", "term": "중기", "price": s2["price"], "pct": s2["dist"], "note": f"과거 {s2['touches']}번 반등한 가격대"})
    if lv and lv.get("last_peak") and lv.get("depth_p75") == lv.get("depth_p75") and lv.get("depth_p75") is not None:
        p = lv["last_peak"] * (1 + lv["depth_p75"] / 100)
        if p < price:
            downs.append({"label": "과거 큰 되돌림 수준", "term": "중기", "price": p, "pct": (p / price - 1) * 100, "note": f"이 종목이 큰 조정에서 {lv['depth_p75']:.0f}%까지 밀린 적이 있어요"})
    if ind and ind.get("lo") and ind["lo"] < price:
        downs.append({"label": "52주 저점", "term": "극단", "price": ind["lo"], "pct": (ind["lo"] / price - 1) * 100, "note": "최근 1년 최저 종가"})
    by_term = lambda t: [d for d in downs if d["term"] == t]
    short = by_term("단기")[0] if by_term("단기") else None
    mid_c = by_term("중기") + ([short] if short else [])
    mid = min(mid_c, key=lambda d: d["price"]) if mid_c else None
    extreme = by_term("극단")[0] if by_term("극단") else None
    base = mid or short or extreme                               # 손익비에 쓰는 보수적 하락 위험
    down_pct = abs(base["pct"]) if base else None
    ratio = (up / down_pct) if (up is not None and down_pct) else None
    ratio_short = (up / abs(short["pct"])) if (up is not None and short and short["pct"]) else None
    need = (down_pct / (up + down_pct) * 100) if (up is not None and down_pct and (up + down_pct) > 0) else None
    # ---- 시나리오 표 (내 평단 기준)
    scen = []
    for name, p in (("상승(보수적)", up_price), ("단기 하락", short["price"] if short else None), ("중기 하락", mid["price"] if mid else None),
                    ("극단 하락", extreme["price"] if extreme else None)):
        if p is None:
            continue
        scen.append({"시나리오": name, "가격": p, "현재가 대비(%)": (p / price - 1) * 100,
                     "내 평단 대비(%)": ((p / avg - 1) * 100) if avg else np.nan,
                     "내 손익 변화(원)": (qty * (p - price)) if qty else np.nan,
                     "총자산 영향(%)": (qty * (p - price) / total_assets * 100) if (qty and total_assets) else np.nan})
    # ---- 계획 이탈 경고
    ret = ((price / avg - 1) * 100) if avg else None
    if ret is not None and rules:
        R = rules.get(tag, {})
        if tag == "스윙" and R.get("손절선") is not None and ret <= R["손절선"]:
            warns.append(f"스윙 종목인데 손절선({R['손절선']:.0f}%)을 넘어 {ret:.1f}%로 보유 중이에요. 처음 계획에서 벗어난 상태예요. 중장기로 바꿨다면 꼬리표를 바꾸고 새 이유를 기록하세요.")
        if tag == "중장기" and R.get("점검선") is not None and ret <= R["점검선"]:
            warns.append(f"점검선({R['점검선']:.0f}%) 아래({ret:.1f}%)예요. 산 이유가 아직 유효한지 먼저 확인하세요.")
    if up is not None and up <= 0:
        warns.append("보수적 상승 여력이 0이에요. 이 근거로는 오를 이유를 찾지 못했어요.")
    if safety_counts and safety_counts.get("fail"):
        warns.append(f"안전 기준 미달이 {safety_counts['fail']}개 있어요. 하락 위험이 위 숫자보다 클 수 있어요.")
    if flow and flow.get("f5") is not None and flow["f5"] < 0 and flow.get("i5", 0) < 0:
        warns.append(f"최근 5일 외국인·기관이 함께 순매도 중이에요(외국인 {flow['f5']:+,.0f}억, 기관 {flow['i5']:+,.0f}억).")
    # ---- 근거 충족도
    have = {"가격 흐름": bool(lv), "컨센서스": bool(cons and cons.get("has")), "수급": bool(flow), "안전·재무": safety_counts is not None}
    # ---- 무효가격(손실 한도) 제안: 가까운 지지선 아래
    invalid = short["price"] * 0.98 if short else None
    return {"price": price, "avg": avg, "qty": qty, "tag": tag, "up": up, "up_price": up_price, "ups": ups, "downs": downs,
            "short": short, "mid": mid, "extreme": extreme, "base": base, "ratio": ratio, "ratio_short": ratio_short,
            "grade": grade(ratio), "need_win": need, "scen": scen, "warns": warns, "have": have, "n_have": sum(have.values()),
            "invalid": invalid, "ret": ret}
