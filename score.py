"""종합 점수: 가격 흐름, 수급, 재무·안전, 컨센서스, 진입 구조(손익비)를 하나의 점수로 모은다.
점수는 '이길 확률'이 아니라 **근거들이 얼마나 한 방향으로 모이는지**를 요약한 것이다. 가중치와 점수 기준은 검증 전이므로
판단 기록·점수 기록으로 실제 성과와 비교해서 고쳐 나간다. 항목마다 가설 실험실의 검증 결과를 함께 보여준다."""
VERSION = 1

import numpy as np

FAMILIES = ["가격 흐름", "수급", "재무·안전", "컨센서스", "진입 구조"]
DEFAULT_WEIGHTS = {"가격 흐름": 25, "수급": 25, "재무·안전": 25, "컨센서스": 15, "진입 구조": 10}
# 점수 항목 -> 가설 실험실 결과 이름(검증 상태를 옆에 보여준다)
ITEM_TO_HYP = {"추세(20·60일선)": "60일선 위", "3개월 수익률": "요인: 6개월 수익률(모멘텀)", "52주 위치": "요인: 52주 고점 대비 위치",
               "큰손(외국인+기관) 20일 순매수 비중": "수급: 큰손(외국인+기관) 20일 순매수 비중", "개인 주도 경고": "수급: 개인 20일 순매수 비중(개인 주도 가설)"}
BANDS = [(65, "근거가 한 방향으로 모여요"), (55, "약하게 우호적"), (45, "중립(평범하거나 엇갈려요)"), (35, "약하게 신중"), (-1, "신중(불리한 근거가 많아요)")]


def _c(x):
    return float(np.clip(x, -100, 100))


def _it(name, score, note):
    return {"항목": name, "점수": _c(score), "설명": note}


def band(total):
    if total is None:
        return "근거 부족"
    return next(label for lim, label in BANDS if total >= lim)


def bucket(total):
    if total is None or total != total:
        return "-"
    return "65 이상" if total >= 65 else "55~65" if total >= 55 else "45~55" if total >= 45 else "35~45" if total >= 35 else "35 미만"


def price_items(ind):
    if not ind:
        return []
    t = (1 if ind["price"] > ind["ma20"] else -1) + (1 if ind["price"] > ind["ma60"] else -1) + (1 if ind["ma20"] > ind["ma60"] else -1)
    out = [_it("추세(20·60일선)", t / 3 * 100, f"종가가 20일선 {'위' if ind['price'] > ind['ma20'] else '아래'}, 60일선 {'위' if ind['price'] > ind['ma60'] else '아래'}, 20일선이 60일선 {'위' if ind['ma20'] > ind['ma60'] else '아래'}")]
    out.append(_it("3개월 수익률", ind["m3"] / 30 * 100, f"{ind['m3']:+.1f}% (±30%에서 만점)"))
    out.append(_it("52주 위치", (ind["pos"] - 50) * 2, f"{ind['pos']:.0f}% (1년 저점 0 ~ 고점 100)"))
    return out


def flow_items(sm):
    if not sm:
        return []

    def share(a, b, c):
        d = abs(a) + abs(b) + abs(c)
        return (a + b) / d if d > 0 else 0.0

    r20, r5 = share(sm["f20"], sm["i20"], sm["p20"]), share(sm["f5"], sm["i5"], sm["p5"])
    out = [_it("큰손(외국인+기관) 20일 순매수 비중", 100 * r20, f"외국인 {sm['f20']:+,.0f}억 · 기관 {sm['i20']:+,.0f}억 · 개인 {sm['p20']:+,.0f}억"),
           _it("큰손 5일 흐름", 100 * r5, f"외국인 {sm['f5']:+,.0f}억 · 기관 {sm['i5']:+,.0f}억 · 개인 {sm['p5']:+,.0f}억")]
    if sm["p20"] > 0 and sm["f20"] < 0 and sm["i20"] < 0:
        out.append(_it("개인 주도 경고", -80, "20일간 개인만 순매수하고 외국인·기관은 순매도(가설: 중장기에는 위험)"))
    elif sm["p20"] > 0 and (sm["f20"] + sm["i20"]) < 0:
        out.append(_it("개인 주도 경고", -40, "20일간 개인이 순매수하고 외국인+기관 합은 순매도(가설: 중장기에는 위험)"))
    if sm.get("hold_chg20") is not None:
        out.append(_it("외국인 보유율 20일 변화", sm["hold_chg20"] * 40, f"{sm['hold_chg20']:+.2f}%p"))
    return out


def fin_items(m, items):
    out = []
    if items:
        cnt = {"fail": 0, "unknown": 0}
        for i in items:
            if i["status"] in cnt:
                cnt[i["status"]] += 1
        out.append(_it("안전 기준", 60 - 40 * cnt["fail"] - 5 * cnt["unknown"], f"미달 {cnt['fail']}개, 확인 불가 {cnt['unknown']}개"))
    if m:
        op = m.get("op3") or []
        if len(op) >= 2 and op[-1] is not None and op[-2] is not None:
            if op[-1] < 0:
                out.append(_it("영업이익 흐름", -60, "최근 영업이익이 적자"))
            else:
                g = (op[-1] - op[-2]) / abs(op[-2]) if op[-2] else 0.0
                out.append(_it("영업이익 흐름", np.clip(g, -1, 1) * 50, f"전년 대비 {g * 100:+.0f}%"))
        if m.get("debt") is not None:
            d = m["debt"]
            out.append(_it("부채비율", 40 if d <= 100 else 10 if d <= 200 else -40, f"{d:,.0f}%"))
        if m.get("roe") is not None:
            r = m["roe"]
            out.append(_it("ROE", 40 if r >= 10 else 15 if r >= 5 else 0 if r >= 0 else -40, f"{r:.1f}%"))
        if m.get("cover") is not None:
            c = m["cover"]
            out.append(_it("이자보상배율", 30 if c >= 5 else 10 if c >= 3 else -10 if c >= 1.5 else -60, "이자 없음" if c >= 999 else f"{c:.1f}배"))
    return out


def cons_items(cons, price, chg=None):
    if not cons or not cons.get("has") or not cons.get("target") or not price:
        return []
    up = (cons["target"] / price - 1) * 100
    out = [_it("목표가 여력(낙관 편향 보정)", (up - 10) * 3, f"평균 목표가 {cons['target']:,.0f}원({up:+.0f}%). 증권사 목표가는 평균 +10%쯤 낙관적이라고 보고 뺐어요")]
    if cons.get("score") is not None:
        out.append(_it("투자의견 평균", (cons["score"] - 4.0) * 60, f"{cons['score']:.2f}점(5점 만점). 매수 의견이 대부분이라 4점을 기준으로 봐요"))
    if chg:
        out.append(_it("목표가 변화", chg[1] * 5, f"{chg[0]}일 전 기록 대비 {chg[1]:+.1f}%"))
    return out


def struct_items(card):
    if not card or card.get("ratio") is None:
        return []
    r = card["ratio"]
    s = float(np.interp(r, [0, 0.3, 0.7, 1.2, 2, 3], [-100, -80, -30, 10, 60, 100]))
    need = card.get("need_win")
    return [_it("손익비(보수적)", s, f"{r:.2f}" + ("" if need is None else f" · 필요 승률 {need:.0f}%"))]


def compute(ind, flow, fin_m, fin_it, cons, cons_chg, card, weights=None, evidence=None):
    """evidence: 항목 이름 -> 검증 결과 문구를 돌려주는 함수(없으면 '미검증')."""
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    price = ind["price"] if ind else None
    fam_items = {"가격 흐름": price_items(ind), "수급": flow_items(flow), "재무·안전": fin_items(fin_m, fin_it),
                 "컨센서스": cons_items(cons, price, cons_chg), "진입 구조": struct_items(card)}
    fams, num, den = [], 0.0, 0.0
    for name in FAMILIES:
        items = fam_items[name]
        for it in items:
            hyp = ITEM_TO_HYP.get(it["항목"])
            it["검증"] = (evidence(hyp) if (evidence and hyp) else ("미검증" if not hyp else "미검증"))
        sc = float(np.mean([i["점수"] for i in items])) if items else None
        fams.append({"name": name, "weight": float(w.get(name, 0)), "score": sc, "items": items})
        if sc is not None and w.get(name, 0) > 0:
            num += w[name] * sc
            den += w[name]
    comp = num / den if den > 0 else None
    # 근거가 적을수록 50점(중립) 쪽으로 당겨서, 한두 가지 근거만으로 점수가 극단으로 나오지 않게 한다
    coverage = (den / sum(w.get(n, 0) for n in FAMILIES)) if sum(w.get(n, 0) for n in FAMILIES) > 0 else 0.0
    total = None if comp is None else 50 + comp / 2 * (coverage ** 0.5)
    avail = [f for f in fams if f["score"] is not None]
    agree = None
    conflicts = []
    if comp is not None and avail:
        strong = [f for f in avail if abs(f["score"]) >= 10]
        agree = (sum(1 for f in strong if np.sign(f["score"]) == np.sign(comp)), len(strong)) if strong else (0, 0)
        pos = [f for f in avail if f["score"] >= 25]
        neg = [f for f in avail if f["score"] <= -25]
        for a in pos:
            for b in neg:
                conflicts.append(f"{a['name']}은(는) 우호적({a['score']:+.0f})인데 {b['name']}은(는) 불리({b['score']:+.0f})해요")
    label = band(total) if len(avail) >= 3 else ("근거 부족" if total is None else "근거 부족(점수는 참고만)")
    return {"total": total, "band": label, "families": fams, "n_avail": len(avail), "n_total": len(FAMILIES), "agree": agree, "conflicts": conflicts,
            "weights": w, "missing": [f["name"] for f in fams if f["score"] is None]}
