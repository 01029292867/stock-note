"""종목 발굴 보강: 규모 구분, 소형주(소액 후보) 기준, 업종별 강도, 진입 스캔 설명.
점수·손익비는 app의 종합 점수와 진입 평가를 그대로 쓴다. 여기서는 그 위에 얹는 계산만 한다."""
VERSION = 1

import numpy as np
import pandas as pd

SMALL_DEFAULT = {"on": False, "cap_min": 0.3, "loss": 0, "debt": 100.0, "roe": 10.0, "growth": True, "pos": 2.0}


def size_class(cap_jo):
    if cap_jo is None or cap_jo != cap_jo:
        return "-"
    return "대형" if cap_jo >= 5 else "중형" if cap_jo >= 1 else "소형"


def small_strict(tbl, cfg):
    """소형주는 변동과 위험이 커서 재무 기준을 더 엄격하게 본다."""
    ok = tbl["영업적자횟수"] <= cfg["loss"]
    ok &= (tbl["부채비율"] <= cfg["debt"]) | (tbl["금융업의심"] & tbl["재무"])
    ok &= tbl["ROE"] >= cfg["roe"]
    if cfg.get("growth", True):
        ok &= tbl["영업이익증가율"] > 0
    return ok.fillna(False)


def sector_strength(closes, sector, min_n=3):
    """업종별로 소속 종목들의 1주·1개월·3개월 수익률 중간값을 구해 강한 순으로 줄 세운다.
    closes: {종목코드: 종가 Series}, sector: {종목코드: 업종 이름}"""
    rows = {}
    for code, s in closes.items():
        sec = sector.get(code)
        if not sec or s is None:
            continue
        c = pd.Series(s).dropna().values
        if len(c) < 65:
            continue
        rows.setdefault(sec, []).append(((c[-1] / c[-6] - 1) * 100, (c[-1] / c[-21] - 1) * 100, (c[-1] / c[-64] - 1) * 100))
    out = []
    for sec, v in rows.items():
        if len(v) < min_n:
            continue
        a = np.array(v)
        out.append({"업종": sec, "종목수": len(v), "1주(%)": float(np.median(a[:, 0])), "1개월(%)": float(np.median(a[:, 1])),
                    "3개월(%)": float(np.median(a[:, 2])), "1개월 상승 비율(%)": float((a[:, 1] > 0).mean() * 100)})
    df = pd.DataFrame(out, columns=["업종", "종목수", "1주(%)", "1개월(%)", "3개월(%)", "1개월 상승 비율(%)"])
    if df.empty:
        return df
    df = df.sort_values("1개월(%)", ascending=False).reset_index(drop=True)
    df["순위"] = np.arange(1, len(df) + 1)
    return df


def attach_sector(df, sector, strength):
    d = df.copy()
    d["업종"] = d["종목코드"].map(sector).fillna("-")
    if strength is not None and len(strength):
        sm = strength.set_index("업종")
        d["업종 1개월(%)"] = d["업종"].map(sm["1개월(%)"])
        d["업종 순위"] = d["업종"].map(sm["순위"])
    else:
        d["업종 1개월(%)"], d["업종 순위"] = np.nan, np.nan
    return d


def reasons(r, n_sectors=None, small_pos=None):
    """진입 후보의 이유를 짧은 말로. r: dict(진입 손익비, 지지선까지(%), 이격도, 흐름, 업종, 업종 1개월(%), 업종 순위, 규모, 52주 위치(%) ...)"""
    out = []
    ratio = r.get("진입 손익비")
    if ratio == ratio and ratio is not None:
        out.append(f"손익비 {ratio:.1f}")
    sup = r.get("지지선까지(%)")
    if sup == sup and sup is not None and abs(sup) <= 8:
        out.append(f"지지선 가까움({sup:+.0f}%)")
    gap = r.get("이격도")
    if gap == gap and gap is not None:
        out.append("과열 주의(이격도 %.0f)" % gap if gap >= 110 else "과열 아님")
    if r.get("흐름") in ("상승 흐름", "조정 중"):
        out.append(r["흐름"])
    rank, s1m = r.get("업종 순위"), r.get("업종 1개월(%)")
    if rank == rank and rank is not None and s1m == s1m:
        tail = f"{r.get('업종')} 1개월 {s1m:+.0f}%, {int(rank)}위" + (f"/{n_sectors}" if n_sectors else "")
        top = max(3, int(round((n_sectors or 0) * 0.3)))
        if s1m > 0 and rank <= top:
            out.append(f"업종 강함({tail})")
        elif s1m < 0:
            out.append(f"업종 약함({tail})")
        else:
            out.append(f"업종 보통({tail})")
    if r.get("규모") == "소형" and small_pos:
        out.append(f"소형주: 비중 {small_pos:g}% 이내로")
    return " · ".join(out)


def criteria_rows(f, risk, small=None):
    """'현재 발굴 기준'을 단계별 표로. f: 발굴 조건, risk: 위험 한도, small: 소형주 설정"""
    def on(k):
        return f[k]["on"]
    r1 = []
    if on("cap"):
        r1.append(f"시가총액 {f['cap']['v']:g}조원 이상")
    if on("tv"):
        r1.append(f"일 거래대금 {f['tv']['v']:g}억원 이상")
    r1.append(f"스팩·우선주 제외, 시가총액 큰 순 최대 {int(f['maxn'])}종목")
    r2 = []
    if on("profit"):
        r2.append("최근 영업이익 흑자")
    if on("loss"):
        r2.append(f"최근 3년 영업적자 {f['loss']['v']}번 이하")
    if on("debt"):
        r2.append(f"부채비율 {f['debt']['v']:g}% 이하(금융업 제외)")
    if on("roe"):
        r2.append(f"ROE {f['roe']['v']:g}% 이상")
    if on("per"):
        r2.append(f"PER {f['per']['v']:g}배 이하")
    if on("pbr"):
        r2.append(f"PBR {f['pbr']['v']:g}배 이하")
    if on("growth"):
        r2.append("영업이익이 전년보다 증가")
    rows = [("① 목록", " · ".join(r1), "크고 거래가 활발한 종목만 남겨요"),
            ("② 재무", " · ".join(r2) or "(조건 없음)", "돈을 벌고 빚이 과하지 않은 회사를 남겨요")]
    if small and small.get("on"):
        rows.append(("② 소형주", f"시가총액 {small['cap_min']:g}조원 이상 {f['cap']['v']:g}조원 미만이면 더 엄격하게: 영업적자 {small['loss']}번 이하 · 부채비율 {small['debt']:g}% 이하 · ROE {small['roe']:g}% 이상"
                     + (" · 영업이익 증가" if small.get("growth", True) else ""), f"통과해도 '소액 후보'로 총자산의 {small['pos']:g}% 이내로만"))
    rows.append(("③ 진입 스캔", f"지금 가격에서 새로 산다면: 보수적 손익비 {risk.get('min_ratio', 2.0):g} 이상이면 '유리', 1.2 이상이면 '보통'. 틀렸을 때 총자산의 {risk.get('per_trade', 1.0):g}%만 잃는 크기로, 한 종목 최대 {risk.get('max_pos', 15.0):g}%",
                 "종합 점수(가격 흐름·재무·수급·컨센서스·진입 구조)로 줄 세우고, 강한 업종·지지선 근처·과열 아님을 같이 보여줘요"))
    rows.append(("④ 상세 점검", "안전 기준 13개(DART 공시) + 수급 + 컨센서스", "사기 직전에 마지막으로 확인해요"))
    return rows
