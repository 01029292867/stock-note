"""목적 자금 계산기: 목표 금액에 닿으려면 연 몇 %의 수익이 필요한지, 시나리오별로 얼마가 되는지 계산한다.
수익은 보장되지 않는다. 여기 숫자는 '필요한 조건'을 보여주는 것이고 예측이 아니다."""
VERSION = 1


def fv(current, monthly, annual_r, years):
    """현재 금액과 매달 저축을 연 수익률로 굴렸을 때 years년 뒤 금액(월 복리)."""
    n = int(round(years * 12))
    if n <= 0:
        return float(current)
    m = (1 + annual_r) ** (1 / 12) - 1
    if abs(m) < 1e-12:
        return float(current + monthly * n)
    return float(current * (1 + m) ** n + monthly * (((1 + m) ** n - 1) / m))


def required_return(target, current, monthly, years, lo=-0.5, hi=2.0):
    """목표에 닿는 데 필요한 연 수익률. 수익률 0%로도 닿으면 0 이하 값을, 불가능하면 None."""
    if years <= 0:
        return None
    if fv(current, monthly, hi, years) < target:
        return None
    if fv(current, monthly, lo, years) >= target:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        if fv(current, monthly, mid, years) >= target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def required_monthly(target, current, annual_r, years):
    """주어진 수익률에서 목표에 닿으려면 매달 저축해야 하는 금액(0 이상)."""
    n = int(round(years * 12))
    if n <= 0:
        return None
    m = (1 + annual_r) ** (1 / 12) - 1
    base = current * (1 + m) ** n
    if target <= base:
        return 0.0
    if abs(m) < 1e-12:
        return (target - current) / n
    return (target - base) * m / ((1 + m) ** n - 1)


def future_target(target, inflation, years):
    """오늘 기준 목표 금액을 물가 상승만큼 키운 금액(years년 뒤에 필요한 명목 금액)."""
    return target * (1 + inflation) ** years


def level(r):
    """필요 수익률의 부담 정도를 말로."""
    if r is None:
        return "비현실적", "지금 조건으로는 연 200%로도 닿지 않아요. 목표 금액, 기간, 매달 저축액을 조정해야 해요."
    if r <= 0:
        return "이미 충분", "수익률 0%로도 목표에 닿아요. 이 돈은 지키는 데 집중해도 되는 구간이에요."
    if r <= 0.03:
        return "부담 낮음", "예금·채권 수준의 수익으로도 가능한 구간이에요. 큰 위험을 질 이유가 크지 않아요."
    if r <= 0.06:
        return "보통", "분산 투자로 노려볼 수 있는 수준이에요. 다만 해마다 손실이 날 수도 있어서 목표 시점이 가까워지면 위험을 줄이는 계획이 필요해요."
    if r <= 0.10:
        return "높은 편", "변동이 큰 자산의 비중이 커야 하는 수준이에요. 목표 시점 직전에 손실이 나면 만회할 시간이 짧아서 부담이 커요."
    return "비현실적일 수 있음", "꾸준히 이 수익을 내기는 매우 어려워요. 수익률을 높이려 하기보다 저축액, 기간, 목표 금액을 먼저 조정하는 게 안전해요."


SCENARIOS = (0.03, 0.05, 0.07)
