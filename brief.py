"""주간 브리핑: 흩어진 점수·신호·계획·기록을 모아서 '이번 주에 먼저 볼 것'을 우선순위로 정리한다.
판단은 사람이 한다. 여기서는 어디부터 볼지, 무엇이 밀려 있는지만 알려준다."""
VERSION = 1

import numpy as np

P_LABEL = {1: "먼저 확인", 2: "이번 주 안에", 3: "여유 있을 때"}
SELL_CHECK = "매도·비중 축소 검토"
PROFIT_CHECK = "수익실현 검토"


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def build(holds, cands, hygiene, port, score_drop=8.0, low_score=35.0):
    """holds: 보유 종목 목록(dict), cands: 매수 후보 목록(dict), hygiene: 기록 상태(dict), port: 포트폴리오 요약(dict)."""
    actions = []
    for h in holds:
        name, w = h["name"], h.get("weight") or 0.0

        def add(p, text, where):
            actions.append({"우선순위": p, "구분": P_LABEL[p], "종목": name, "내용": text, "볼 곳": where, "비중(%)": w})

        if h.get("plan_level") == "warn":
            add(1, f"매매 계획 기준 이탈 — {h['plan_action']}", "매매 계획")
        if h.get("verdict") == SELL_CHECK:
            sig = ", ".join(h.get("signals") or []) or "신호 없음"
            add(1, f"판정이 '{SELL_CHECK}'예요({sig}). 산 이유가 아직 유효한지 점검하세요", "종목 리포트")
        sc, prev, n = _num(h.get("score")), _num(h.get("score_prev")), h.get("n_avail", 0)
        if sc is not None and n >= 3 and sc <= low_score:
            add(2, f"종합 점수가 낮아요({sc:.0f}점, 근거 {n}/5)", "종목 리포트")
        if sc is not None and prev is not None and n >= 3 and sc - prev <= -score_drop:
            add(2, f"종합 점수가 지난주보다 {abs(sc - prev):.0f}점 떨어졌어요({prev:.0f} → {sc:.0f})", "종목 리포트")
        if h.get("plan_level") == "info":
            add(2, f"매매 계획 — {h['plan_action']}", "매매 계획")
        flags = h.get("flags") or []
        if "개인 주도" in flags:
            add(2, "최근 20일 개인만 순매수하고 외국인·기관은 순매도 중이에요(개인 주도 경고)", "종목 리포트")
        if "안전 미달" in flags:
            add(2, "안전 기준 미달 항목이 있어요", "안전 점검")
        if h.get("verdict") == PROFIT_CHECK:
            add(3, f"수익이 목표를 넘었어요({h['ret']:+.0f}%). 이익 일부 확정이나 보호선 상향을 점검하세요" if _num(h.get("ret")) is not None else "수익이 목표를 넘었어요", "종목 리포트")
        if any(f.startswith("근거 엇갈림") for f in flags):
            add(3, "근거 종류끼리 방향이 엇갈려요(" + next(f for f in flags if f.startswith("근거 엇갈림")) + ")", "종목 리포트")
    # 기록·자료 상태
    if hygiene.get("due_n"):
        actions.append({"우선순위": 2, "구분": P_LABEL[2], "종목": "(전체)", "내용": f"30일 넘게 판단 기록이 없는 보유 종목이 {hygiene['due_n']}개예요(월간 점검)", "볼 곳": "판단 기록", "비중(%)": 0.0})
    if hygiene.get("flowlog_days") is not None and hygiene["flowlog_days"] >= 25:
        actions.append({"우선순위": 3, "구분": P_LABEL[3], "종목": "(전체)", "내용": f"수급 기록이 {hygiene['flowlog_days']}일째 비어 있어요. 한 달 넘게 비면 빈틈이 생겨요", "볼 곳": "가설 실험실", "비중(%)": 0.0})
    elif hygiene.get("flowlog_days") is None:
        actions.append({"우선순위": 3, "구분": P_LABEL[3], "종목": "(전체)", "내용": "수급 기록을 아직 시작하지 않았어요. 검증 자료는 오늘부터 쌓여요", "볼 곳": "가설 실험실", "비중(%)": 0.0})
    if hygiene.get("cash_days") is not None and hygiene["cash_days"] >= 35:
        actions.append({"우선순위": 3, "구분": P_LABEL[3], "종목": "(전체)", "내용": f"현금 기록이 {hygiene['cash_days']}일 전이에요. 수익률 계산을 위해 갱신하세요(월말)", "볼 곳": "목표·성과", "비중(%)": 0.0})
    if hygiene.get("stale_fin_n"):
        actions.append({"우선순위": 3, "구분": P_LABEL[3], "종목": "(전체)", "내용": f"재무 자료가 없거나 오래된 보유 종목이 {hygiene['stale_fin_n']}개예요", "볼 곳": "안전 점검", "비중(%)": 0.0})
    if hygiene.get("missing_price_n"):
        actions.append({"우선순위": 2, "구분": P_LABEL[2], "종목": "(전체)", "내용": f"시세를 가져오지 못한 종목이 {hygiene['missing_price_n']}개예요. 점수와 평가가 빠져 있을 수 있어요", "볼 곳": "내 자산", "비중(%)": 0.0})
    # 포트폴리오 점검
    if port.get("top_weight") is not None and port.get("max_pos") and port["top_weight"] > port["max_pos"]:
        actions.append({"우선순위": 2, "구분": P_LABEL[2], "종목": port["top_name"], "내용": f"한 종목 비중이 {port['top_weight']:.0f}%로 정해둔 최대 비중({port['max_pos']:.0f}%)을 넘었어요", "볼 곳": "내 자산", "비중(%)": port["top_weight"]})
    actions.sort(key=lambda x: (x["우선순위"], -x["비중(%)"]))
    good = [c for c in cands if c.get("score") is not None and c.get("n_avail", 0) >= 3 and c.get("score") >= 55
            and c.get("entry_verdict") in ("유리", "보통") and c.get("ratio") is not None]
    good.sort(key=lambda c: (-c["score"], -c["ratio"]))
    counts = {p: sum(1 for a in actions if a["우선순위"] == p) for p in (1, 2, 3)}
    head = (f"먼저 확인할 것 {counts[1]}건 · 이번 주 안에 {counts[2]}건 · 여유 있을 때 {counts[3]}건" if actions else "이번 주에 급하게 볼 것은 없어요")
    head += f" · 매수 후보 {min(3, len(good))}개" if good else " · 조건을 모두 만족하는 매수 후보는 없어요"
    return {"headline": head, "counts": counts, "actions": actions, "cands": good[:3], "n_cands_all": len(cands)}


def to_text(res, today):
    lines = [f"[주간 브리핑 {today}]", res["headline"], ""]
    for p in (1, 2, 3):
        rows = [a for a in res["actions"] if a["우선순위"] == p]
        if rows:
            lines.append(f"■ {P_LABEL[p]}")
            lines += [f"- {a['종목']}: {a['내용']} → {a['볼 곳']}" for a in rows]
            lines.append("")
    if res["cands"]:
        lines.append("■ 매수 후보(점수 높고 진입 구조가 괜찮은 관심종목)")
        for c in res["cands"]:
            lines.append(f"- {c['name']}: 종합 {c['score']:.0f}점, 진입 손익비 {c['ratio']:.2f}" + (f", 무효가격 {c['invalid']:,.0f}원" if c.get("invalid") else "")
                         + (f", 최대 {c['shares']:,}주" if c.get("shares") else ""))
    lines.append("")
    lines.append("※ 점검 신호이지 사고팔라는 뜻이 아니에요. 판단은 직접 하세요.")
    return "\n".join(lines)
