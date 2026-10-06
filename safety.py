"""안전 기준 판단. DART 데이터와 시세로 지표를 만들고, 내 기준에 대조한다.
상태: pass(통과) / fail(미달) / unknown(확인 불가) / na(해당 없음)"""
import copy

DEFAULT_SAFE = {
    "cap": {"on": True, "v": 1.0},      # 시가총액(조원) 이상
    "loss": {"on": True, "v": 1},       # 최근 3년 영업적자 횟수 이하
    "cover": {"on": True, "v": 3.0},    # 이자보상배율 이상
    "debt": {"on": True, "v": 200.0},   # 부채비율(%) 이하
    "ocf": {"on": True},                # 영업활동현금흐름 흑자
    "impair": {"on": True},             # 자본잠식 없음
    "audit": {"on": True},              # 감사의견 적정
    "distress": {"on": True},           # 부도·회생·관리절차 공시 없음
    "admin": {"on": True},              # 관리종목 지정 없음(거래소 목록)
    "divy": {"on": True, "v": 3},       # 최근 3년 중 현금배당 N년 이상
    "major": {"on": True, "v": 10.0},   # 최대주주 지분율(%) 이상
    "issue": {"on": True, "v": 0},      # 최근 3년 증자·CB·BW·EB 결정 횟수 이하
    "tv": {"on": True, "v": 30.0},      # 일평균 거래대금(억원) 이상
}
FIN_PREFIX = ("64", "65", "66")  # 금융업(KSIC)


def default_rules():
    return copy.deepcopy(DEFAULT_SAFE)


def derive(d, price=None, tv_eok=None, admin=None):
    """DART 원자료 d(dict)와 시세에서 판단용 지표를 만든다."""
    m = {"is_fin": False, "admin": admin}
    m["is_pref"] = bool((d or {}).get("resolved_code"))
    m["resolved_code"] = (d or {}).get("resolved_code")
    comp = (d or {}).get("company") or {}
    m["name"] = comp.get("name") or ""
    m["is_fin"] = str(comp.get("induty", "")).startswith(FIN_PREFIX)
    fin = (d or {}).get("fin")
    m["fin_year"] = fin.get("year") if fin else None
    m["fs"] = fin.get("fs") if fin else None
    op = fin.get("op") if fin else None
    m["op3"] = op
    known = [x for x in (op or []) if x is not None]
    m["loss_n"] = sum(1 for x in known if x <= 0) if known else None
    m["loss_known"] = len(known)
    eq, li, cap = (fin or {}).get("equity"), (fin or {}).get("liab"), (fin or {}).get("capital")
    m["equity"], m["liab"] = eq, li
    m["debt"] = (li / eq * 100) if (eq and eq > 0 and li is not None) else None
    m["roe"] = ((fin or {}).get("ni") / eq * 100) if (eq and eq > 0 and (fin or {}).get("ni") is not None) else None
    m["ni"] = (fin or {}).get("ni")
    intr = (fin or {}).get("interest")
    m["interest"] = intr
    m["interest_basis"] = (fin or {}).get("interest_basis")
    last_op = op[2] if op and op[2] is not None else None
    if last_op is not None and intr:
        m["cover"] = last_op / intr
    elif last_op is not None and intr == 0:
        m["cover"] = 999.0  # 이자비용이 0이면 이자 부담이 없다고 본다
    else:
        m["cover"] = None
    m["ocf"] = (fin or {}).get("ocf")
    m["impair"] = (eq < cap) if (eq is not None and cap) else None
    div = (d or {}).get("div")
    m["div_list"] = div.get("per_share") if div else None
    m["div_years"] = sum(1 for x in div["per_share"] if x and x > 0) if div else None
    mj = (d or {}).get("major")
    m["major"] = mj.get("ratio") if mj else None
    sh = (d or {}).get("shares")
    m["shares"] = sh.get("common") if sh else None
    # 우선주 가격에 보통주 주식 수를 곱하면 틀려서, 우선주는 시가총액을 계산하지 않는다
    m["cap_jo"] = (price * m["shares"] / 1e12) if (price and m["shares"] and not m["is_pref"]) else None
    au = (d or {}).get("audit")
    m["audit"] = au.get("opinion") if au else None
    if au and au.get("opinion"):
        o = au["opinion"]
        m["audit_ok"] = ("적정" in o) and ("부적정" not in o) and ("한정" not in o)
    else:
        m["audit_ok"] = None
    iss = (d or {}).get("issues")
    m["issue_n"] = sum(iss.values()) if iss else None
    m["issues"] = iss
    m["issue_detail"] = (d or {}).get("issue_detail") or {}
    ds = (d or {}).get("distress")
    m["distress_n"] = sum(ds.values()) if ds else None
    m["tv"] = tv_eok
    return m


def _r(key, label, status, detail):
    return {"key": key, "label": label, "status": status, "detail": detail}


def _issue_text(m):
    parts = []
    for name, lst in (m.get("issue_detail") or {}).items():
        for x in lst:
            dtx = x.get("date", "")
            dtx = f"{dtx[:4]}-{dtx[4:6]}-{dtx[6:8]}" if len(dtx) == 8 else dtx
            parts.append(f"{dtx} {name}" + (f"({x['method']})" if x.get("method") else ""))
    return ", ".join(parts)


def evaluate(m, rules):
    out = []
    f = rules
    fin = m["is_fin"]

    def fmt_eok(x):
        return f"{x / 1e8:,.0f}억"

    if f["cap"]["on"]:
        v = m["cap_jo"]
        lab = f"시가총액 {f['cap']['v']:g}조원 이상"
        out.append(_r("cap", lab, "unknown" if v is None else ("pass" if v >= f["cap"]["v"] else "fail"),
                      ("우선주는 보통주 회사 기준이라 계산하지 않아요" if m.get("is_pref") else "확인 불가") if v is None else f"{v:,.2f}조원"))
    if f["loss"]["on"]:
        n = m["loss_n"]
        lab = f"최근 3년 영업적자 {f['loss']['v']:g}번 이하"
        if n is None or m["loss_known"] < 3:
            if n is not None and n > f["loss"]["v"]:
                out.append(_r("loss", lab, "fail", f"{n}번 (확인된 {m['loss_known']}개 연도 중)"))
            else:
                out.append(_r("loss", lab, "unknown", "영업이익 자료가 부족해요"))
        else:
            out.append(_r("loss", lab, "pass" if n <= f["loss"]["v"] else "fail", f"{n}번"))
    if f["cover"]["on"]:
        v = m["cover"]
        lab = f"이자보상배율 {f['cover']['v']:g}배 이상"
        if fin:
            out.append(_r("cover", lab, "na", "금융업은 해당 없음"))
        elif v is None:
            out.append(_r("cover", lab, "unknown", "이자비용 확인 불가"))
        else:
            basis = f" ({m['interest_basis']} 기준)" if m.get("interest_basis") and m["interest_basis"] != "이자비용" else ""
            txt = "이자 부담 없음" if v >= 999 else f"{v:,.1f}배{basis}"
            out.append(_r("cover", lab, "pass" if v >= f["cover"]["v"] else "fail", txt))
    if f["debt"]["on"]:
        v = m["debt"]
        lab = f"부채비율 {f['debt']['v']:g}% 이하"
        if fin:
            out.append(_r("debt", lab, "na", "금융업은 해당 없음"))
        elif v is None:
            out.append(_r("debt", lab, "fail" if (m["equity"] is not None and m["equity"] <= 0) else "unknown",
                          "자본총계가 0 이하" if (m["equity"] is not None and m["equity"] <= 0) else "확인 불가"))
        else:
            out.append(_r("debt", lab, "pass" if v <= f["debt"]["v"] else "fail", f"{v:,.0f}%"))
    if f["ocf"]["on"]:
        v = m["ocf"]
        lab = "영업활동현금흐름 흑자"
        if fin:
            out.append(_r("ocf", lab, "na", "금융업은 해당 없음"))
        elif v is None:
            out.append(_r("ocf", lab, "unknown", "확인 불가"))
        else:
            out.append(_r("ocf", lab, "pass" if v > 0 else "fail", ("+" if v >= 0 else "-") + fmt_eok(abs(v))))
    if f["impair"]["on"]:
        v = m["impair"]
        out.append(_r("impair", "자본잠식 없음", "unknown" if v is None else ("fail" if v else "pass"),
                      "확인 불가" if v is None else ("자본잠식(자본총계 < 자본금)" if v else "없음")))
    if f["audit"]["on"]:
        v = m["audit_ok"]
        out.append(_r("audit", "감사의견 적정", "unknown" if v is None else ("pass" if v else "fail"),
                      "확인 불가" if v is None else (m["audit"] or "")))
    if f["distress"]["on"]:
        v = m["distress_n"]
        out.append(_r("distress", "부도·회생절차·관리절차 공시 없음(최근 3년)", "unknown" if v is None else ("pass" if v == 0 else "fail"),
                      "확인 불가" if v is None else f"{v}건"))
    if f["admin"]["on"]:
        v = m.get("admin")
        out.append(_r("admin", "관리종목 지정 없음", "unknown" if v is None else ("fail" if v else "pass"),
                      "확인 불가(거래소 목록을 못 불러왔어요)" if v is None else ("관리종목으로 지정돼 있어요" if v else "없음")))
    if f["divy"]["on"]:
        v = m["div_years"]
        lab = f"최근 3년 중 현금배당 {f['divy']['v']:g}년 이상"
        out.append(_r("divy", lab, "unknown" if v is None else ("pass" if v >= f["divy"]["v"] else "fail"),
                      "확인 불가" if v is None else f"3년 중 {v}년"))
    if f["major"]["on"]:
        v = m["major"]
        lab = f"최대주주 지분율 {f['major']['v']:g}% 이상"
        if fin and v is not None and v < f["major"]["v"]:
            out.append(_r("major", lab, "na", f"{v:.1f}% (지배주주가 없는 금융지주 등은 낮을 수 있어요)"))
        else:
            out.append(_r("major", lab, "unknown" if v is None else ("pass" if v >= f["major"]["v"] else "fail"),
                          "확인 불가" if v is None else f"{v:.1f}%"))
    if f["issue"]["on"]:
        v = m["issue_n"]
        lab = f"최근 3년 유상증자·전환사채 등 발행 결정 {f['issue']['v']:g}번 이하"
        out.append(_r("issue", lab, "unknown" if v is None else ("pass" if v <= f["issue"]["v"] else "fail"),
                      "확인 불가" if v is None else (f"{v}건" + (f": {_issue_text(m)}" if _issue_text(m) else ""))))
    if f["tv"]["on"]:
        v = m["tv"]
        lab = f"일평균 거래대금 {f['tv']['v']:g}억원 이상"
        out.append(_r("tv", lab, "unknown" if v is None else ("pass" if v >= f["tv"]["v"] else "fail"),
                      "확인 불가" if v is None else f"{v:,.0f}억원"))
    return out


def summarize(items):
    c = {"pass": 0, "fail": 0, "unknown": 0, "na": 0}
    for i in items:
        c[i["status"]] += 1
    return c


def headline(items):
    c = summarize(items)
    if c["fail"]:
        return f"미달 {c['fail']}개"
    if c["unknown"]:
        return f"통과 (확인 불가 {c['unknown']}개)"
    return "모두 통과"
