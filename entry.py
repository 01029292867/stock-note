"""매수 후보 평가: 살 때의 구조(손익비, 무효가격, 포지션 크기)를 정리한다.
'잃지 않는 종목'을 찾는 게 아니라, 틀렸을 때 잃는 크기를 미리 정해두는 도구다."""
VERSION = 1

import math

DEFAULTS = {"per_trade": 1.0, "max_pos": 15.0, "min_ratio": 2.0}


def plan_entry(price, card, total_assets, risk=None, invalid=None):
    """card: judge.build_card 결과. risk: {'per_trade': 한 번 틀렸을 때 총자산 대비 손실 한도(%), 'max_pos': 한 종목 최대 비중(%), 'min_ratio': 진입 손익비 기준}"""
    r = dict(DEFAULTS, **(risk or {}))
    inv = invalid if invalid else (card.get("invalid") if card else None)
    up = card.get("up") if card else None
    up_price = card.get("up_price") if card else None
    out = {"price": price, "invalid": inv, "up": up, "up_price": up_price, "risk": r, "notes": []}
    if not inv or inv >= price:
        out.update({"verdict": "무효가격 필요", "stop_pct": None, "ratio": None, "need": None, "shares": None, "amount": None,
                    "pct_assets": None, "max_loss": None, "wait_price": None})
        out["notes"].append("현재가 아래에서 의미 있는 지지선을 찾지 못했어요. 틀렸다고 인정할 무효가격을 직접 정해야 크기를 계산할 수 있어요.")
        return out
    stop = (price - inv) / price * 100
    ratio = (up / stop) if (up is not None and stop > 0) else None
    need = (stop / (up + stop) * 100) if (up is not None and (up + stop) > 0) else None
    shares = amount = pct = max_loss = None
    capped = False
    if total_assets:
        max_loss = total_assets * r["per_trade"] / 100
        shares = int(math.floor(max_loss / (price - inv)))
        amount = shares * price
        cap = total_assets * r["max_pos"] / 100
        if amount > cap:
            shares, amount, capped = int(math.floor(cap / price)), int(math.floor(cap / price)) * price, True
        pct = amount / total_assets * 100
    wait = None
    if ratio is not None and up_price and ratio < r["min_ratio"]:
        w = (up_price + r["min_ratio"] * inv) / (1 + r["min_ratio"])
        if inv < w < price:
            wait = w
    if ratio is None:
        verdict = "근거 부족"
    elif ratio >= r["min_ratio"]:
        verdict = "유리"
    elif ratio >= 1.2:
        verdict = "보통"
    else:
        verdict = "불리(기다리는 편이 나아요)"
    if capped:
        out["notes"].append(f"한 종목 최대 비중({r['max_pos']:.0f}%)에 걸려서 위험 한도보다 적게 샀어요.")
    out.update({"verdict": verdict, "stop_pct": stop, "ratio": ratio, "need": need, "shares": shares, "amount": amount, "pct_assets": pct,
                "max_loss": max_loss, "wait_price": wait, "wait_pct": ((wait / price - 1) * 100) if wait else None, "capped": capped,
                "tranche": (shares // 3) if shares is not None else None})
    return out


def entry_text(e):
    """쉬운 말 풀이."""
    if e["stop_pct"] is None:
        return " ".join(e["notes"])
    t = (f"**무효가격 {e['invalid']:,.0f}원**은 현재가보다 {e['stop_pct']:.1f}% 아래예요. 여기까지 내려오면 '내 판단이 틀렸다'고 보고 정리하겠다고 미리 정해두는 가격이에요. ")
    if e["ratio"] is not None:
        t += (f"이 가격에서 새로 산다면 오를 폭(보수적으로 +{e['up']:.1f}%) 대 틀렸을 때 잃는 폭(-{e['stop_pct']:.1f}%)이 **{e['ratio']:.2f} : 1**이고, "
              f"본전이 되려면 약 {e['need']:.0f}%를 맞혀야 해요. 기준({e['risk']['min_ratio']:.1f} 이상)과 비교한 판정: **{e['verdict']}**. ")
    if e["wait_price"]:
        t += f"\n\n손익비가 기준에 닿으려면 약 **{e['wait_price']:,.0f}원**(현재가보다 {e['wait_pct']:.1f}%)까지 내려올 때를 기다리는 방법이 있어요(같은 무효가격과 목표 가정). "
    if e["shares"] == 0:
        t += (f"\n\n**크기**: 틀렸을 때 총자산의 {e['risk']['per_trade']:.1f}%(약 {e['max_loss']:,.0f}원)만 잃도록 계산하면 **한 주도 살 수 없어요.** "
              "무효가격이 너무 가깝거나, 한 주 가격에 비해 손실 한도가 작은 경우예요. 손실 한도(규칙 탭)나 무효가격을 조정해 보세요.")
    elif e["shares"] is not None:
        t += (f"\n\n**크기**: 틀렸을 때 총자산의 {e['risk']['per_trade']:.1f}%(약 {e['max_loss']:,.0f}원)만 잃도록 하면 최대 **{e['shares']:,}주(약 {e['amount']:,.0f}원, 총자산의 {e['pct_assets']:.1f}%)**까지 살 수 있어요. "
              f"한 번에 다 사기보다 3번에 나누면 한 번에 약 {e['tranche']:,}주예요.")
    if e["notes"]:
        t += "\n\n" + " ".join(e["notes"])
    return t
