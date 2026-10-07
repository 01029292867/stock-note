"""내 투자 노트 - 2단계: 내 자산 + 안전 점검(DART) + 종목 리포트 (실제 시세, 구글 시트 저장)"""
import copy
import datetime as dt
import hashlib
import hmac
import json
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import consensus
import brief
import dart_data
import discover
import entry
import flows
import fund
import journal
import judge
import lab
import market
import perf
import plan
import reports
import safety
import score
import signals as sg
from store import GasStore

st.set_page_config(page_title="내 투자 노트", page_icon="📈", layout="wide")

BROKERS = ["키움", "한국투자", "카카오", "토스"]
TAGS = ["중장기", "스윙"]
HOLD_COLS = ["증권사", "종목코드", "종목명", "꼬리표", "수량", "평균단가", "매수일"]
KAKAO_COLS = ["종목명", "티커", "하루금액", "횟수", "기준일", "시작일"]
CONC_LIMIT = 20  # 한 종목 쏠림 경고 기준(%)
CACHE_COLS = ["종목코드", "갱신일", "데이터"]
SETTINGS_COLS = ["이름", "값"]
WATCH_COLS = ["종목코드", "종목명"]
SCORELOG_COLS = ["날짜", "종목코드", "종합점수", "가격", "수급", "재무·안전", "컨센서스", "진입 구조", "근거수", "종가"]
HYP_COLS = ["실행일", "가설", "종목수", "기간(년)", "보유(일)", "표본", "승률차(%p)", "평균초과수익(%)", "CI하한", "CI상한", "일관", "등급"]
FLOWLOG_COLS = ["날짜", "종목코드", "기관(주)", "외국인(주)", "개인(주)", "종가", "외국인보유율"]
FUND_COLS = ["이름", "목표(만원)", "기간(년)", "배분(%)", "매달 투입(만원)"]
DEFAULT_FUNDS = [{"이름": "자녀 분가 자금", "목표(만원)": 0.0, "기간(년)": 5.0, "배분(%)": 40.0, "매달 투입(만원)": 0.0},
                 {"이름": "노후 자금", "목표(만원)": 0.0, "기간(년)": 15.0, "배분(%)": 60.0, "매달 투입(만원)": 0.0}]
JOURNAL_DUE_DAYS = 30  # 이 기간 넘게 판단 기록이 없는 보유 종목은 점검 대상으로 안내한다
CONS_COLS = ["날짜", "종목코드", "목표가", "의견점수"]
FRESH_DAYS = 7  # 재무·공시 데이터를 이 기간 안에는 다시 가져오지 않는다
STATUS_ICON = {"pass": "✅", "fail": "❌", "unknown": "❔", "na": "➖"}
SAFE_LABELS = {
    "cap": ("시가총액", "조원 이상"), "loss": ("최근 3년 영업적자", "번 이하"), "cover": ("이자보상배율", "배 이상"),
    "debt": ("부채비율", "% 이하"), "ocf": ("영업활동현금흐름 흑자", None), "impair": ("자본잠식 없음", None),
    "audit": ("감사의견 적정", None), "distress": ("부도·회생·관리절차 공시 없음", None),
    "admin": ("관리종목 지정 없음", None),
    "divy": ("최근 3년 중 현금배당", "년 이상"), "major": ("최대주주 지분율", "% 이상"),
    "issue": ("최근 3년 유상증자·전환사채 등 발행", "번 이하"), "tv": ("일평균 거래대금", "억원 이상"),
}

DEMO_HOLD = pd.DataFrame(
    [
        ["키움", "005930", "삼성전자", "중장기", 10, 60000, ""],
        ["키움", "000660", "SK하이닉스", "스윙", 3, 150000, ""],
        ["한국투자", "005380", "현대차", "중장기", 5, 200000, ""],
    ],
    columns=HOLD_COLS,
)
DEMO_KAKAO = pd.DataFrame([["엔비디아", "NVDA", 3000, None, "", "2026-03-02"]], columns=KAKAO_COLS)


# ---------- 설정/보안 ----------
def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


def gate():
    pw = secret("APP_PASSWORD")
    if not pw:
        if secret("GAS_URL"):
            st.error("보안을 위해 APP_PASSWORD(비밀번호)를 설정해야 내 자산을 보여줘요. README의 3단계를 확인해 주세요.")
            st.stop()
        return
    if st.session_state.get("authed"):
        return
    st.title("📈 내 투자 노트")
    with st.form("login"):
        typed = st.text_input("비밀번호", type="password")
        if st.form_submit_button("들어가기"):
            if hmac.compare_digest(str(typed).encode('utf-8'), str(pw).encode('utf-8')):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("비밀번호가 맞지 않아요.")
    st.stop()


@st.cache_resource
def _make_store(url, token):
    return GasStore(url, token)


def get_store():
    url, token = secret("GAS_URL"), secret("GAS_TOKEN")
    return _make_store(url, token) if url and token else None



# ---------- 규칙 저장 (구글 시트 '설정' 탭) ----------
def _merge(base, saved):
    """저장된 값을 기본값 위에 덮어쓴다. 새로 생긴 항목은 기본값을 유지한다."""
    for k, v in (saved or {}).items():
        if k in base:
            if isinstance(base[k], dict) and isinstance(v, dict):
                _merge(base[k], v)
            else:
                base[k] = v


def load_settings():
    if st.session_state.get("settings_loaded"):
        return
    st.session_state.settings_loaded = True
    store = get_store()
    if store is None or st.session_state.get("load_error"):
        return
    try:
        df = store.read("설정", SETTINGS_COLS)
        saved = {r["이름"]: json.loads(r["값"]) for r in df.to_dict("records") if r.get("이름") in ("rules", "safe", "disc", "goal", "risk", "funds", "weights")}
    except Exception:
        return  # 설정을 못 읽으면 기본값으로 시작한다(저장된 값을 지우지는 않는다)
    _merge(st.session_state.rules, saved.get("rules"))
    _merge(st.session_state.safe, saved.get("safe"))
    _merge(st.session_state.disc_f, saved.get("disc"))
    _merge(st.session_state.goal, saved.get("goal"))
    _merge(st.session_state.risk, saved.get("risk"))
    _merge(st.session_state.weights, saved.get("weights"))
    if isinstance(saved.get("funds"), list) and saved["funds"]:
        st.session_state.funds = saved["funds"]


def save_settings():
    """반환: (성공 여부, 안내 문구)"""
    store = get_store()
    if store is None:
        return True, "임시로 반영했어요(데모 모드라서 저장되지는 않아요)."
    if st.session_state.get("load_error"):
        return False, "구글 시트 연결이 불안정해서 저장하지 않았어요."
    try:
        rows = [["rules", json.dumps(st.session_state.rules, ensure_ascii=False)],
                ["safe", json.dumps(st.session_state.safe, ensure_ascii=False)],
                ["disc", json.dumps(st.session_state.disc_f, ensure_ascii=False)],
                ["goal", json.dumps(st.session_state.goal, ensure_ascii=False)],
                ["risk", json.dumps(st.session_state.risk, ensure_ascii=False)],
                ["weights", json.dumps(st.session_state.weights, ensure_ascii=False)],
                ["funds", json.dumps(st.session_state.funds, ensure_ascii=False)]]
        store.write("설정", pd.DataFrame(rows, columns=SETTINGS_COLS))
        return True, "규칙을 저장했어요. 이제 새로고침하거나 폰에서 열어도 그대로예요."
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"


def reset_settings():
    st.session_state.rules = copy.deepcopy(sg.DEFAULT_RULES)
    st.session_state.safe = safety.default_rules()
    st.session_state.disc_f = copy.deepcopy(discover.DEFAULT_FILTERS)
    st.session_state.goal = copy.deepcopy(perf.DEFAULT_GOAL)
    st.session_state.risk = copy.deepcopy(entry.DEFAULTS)
    st.session_state.weights = copy.deepcopy(score.DEFAULT_WEIGHTS)
    for k in list(st.session_state.keys()):
        if k.startswith(("s_on_", "s_v_", "d_on_", "d_v_", "g_")) or k in ("wt_0", "wt_1", "wt_2", "wt_3", "wt_4", "rk_pt", "rk_mp", "rk_mr", "l_t", "s_t", "d_maxn", "l_ta", "l_tw", "s_ta", "s_tw", "l_td", "s_td", "l_sp", "s_sp"):
            del st.session_state[k]



# ---------- 관심종목 (구글 시트 '관심종목' 탭) ----------
def load_watch():
    if "watch" in st.session_state:
        return
    df = pd.DataFrame(columns=WATCH_COLS)
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            df = store.read("관심종목", WATCH_COLS)
            df["종목코드"] = df["종목코드"].astype(str).str.strip().str.zfill(6)
        except Exception:
            pass
    st.session_state.watch = df.reset_index(drop=True)


def add_watch(items):
    """items: [(코드, 이름)]. 반환: (성공, 문구)"""
    cur = st.session_state.watch
    have = set(cur["종목코드"])
    new = [(c, n) for c, n in items if c not in have]
    if not new:
        return True, "이미 모두 관심종목에 있어요."
    df = pd.concat([cur, pd.DataFrame(new, columns=WATCH_COLS)], ignore_index=True)
    store = get_store()
    try:
        if store is not None:
            if st.session_state.get("load_error"):
                return False, "구글 시트 연결이 불안정해서 저장하지 않았어요."
            store.write("관심종목", df)
        st.session_state.watch = df
        return True, f"{len(new)}개를 관심종목에 추가했어요." + ("" if store is not None else "(데모 모드라서 저장되지는 않아요)")
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"



# ---------- 성과 기록 (구글 시트 '자산기록'·'입출금' 탭) ----------
def snaps_view(df):
    """편집·저장용: 날짜는 글자, 금액은 숫자로 맞춘다."""
    d = df.copy()
    for c in perf.SNAP_COLS:
        if c not in d.columns:
            d[c] = None
    for c in ("주식", "현금", "총자산"):
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "", regex=False).replace({"": None, "None": None, "nan": None}), errors="coerce")
    for c in ("날짜", "구분", "메모"):
        d[c] = d[c].fillna("").astype(str)
    return d[perf.SNAP_COLS].reset_index(drop=True)


def flows_view(df):
    d = df.copy()
    for c in perf.FLOW_COLS:
        if c not in d.columns:
            d[c] = None
    d["금액"] = pd.to_numeric(d["금액"].astype(str).str.replace(",", "", regex=False).replace({"": None, "None": None, "nan": None}), errors="coerce")
    for c in ("날짜", "메모"):
        d[c] = d[c].fillna("").astype(str)
    return d[perf.FLOW_COLS].reset_index(drop=True)


def load_perf():
    if "perf_snaps" in st.session_state:
        return
    snaps, flows = pd.DataFrame(columns=perf.SNAP_COLS), pd.DataFrame(columns=perf.FLOW_COLS)
    err = None
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            snaps = store.read("자산기록", perf.SNAP_COLS)
            flows = store.read("입출금", perf.FLOW_COLS)
        except Exception as e:
            err = str(e)  # 읽지 못한 상태에서 저장하면 기존 기록을 덮어쓰므로 저장을 막는다
    st.session_state.perf_snaps = snaps_view(snaps)
    st.session_state.perf_flows = flows_view(flows)
    st.session_state.perf_err = err


def _to_store_text(df, num_cols):
    d = df.copy()
    for c in num_cols:
        d[c] = d[c].map(lambda x: "" if pd.isna(x) else str(int(round(float(x)))))
    return d


def save_perf(snaps=None, flows=None):
    """반환: (성공, 문구)"""
    store = get_store()
    if store is not None and (st.session_state.get("load_error") or st.session_state.get("perf_err")):
        return False, "구글 시트에서 기록을 읽지 못한 상태라 저장을 막았어요."
    try:
        if snaps is not None:
            c = perf.clean_snaps(snaps)
            c["날짜"] = c["날짜"].dt.strftime("%Y-%m-%d")
            c["구분"] = c["구분"].fillna("").replace("", "수동")
            c["메모"] = c["메모"].fillna("")
            out = _to_store_text(c[perf.SNAP_COLS], ["주식", "현금", "총자산"])
            if store is not None:
                store.write("자산기록", out)
            st.session_state.perf_snaps = snaps_view(out)
        if flows is not None:
            c = perf.clean_flows(flows)
            c["날짜"] = c["날짜"].dt.strftime("%Y-%m-%d")
            c["메모"] = c["메모"].fillna("")
            out = _to_store_text(c[perf.FLOW_COLS], ["금액"])
            if store is not None:
                store.write("입출금", out)
            st.session_state.perf_flows = flows_view(out)
        return True, "저장했어요." if store is not None else "임시로 반영했어요(데모 모드라서 저장되지는 않아요)."
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"



# ---------- 판단 기록 (구글 시트 '판단기록' 탭) ----------
def load_journal():
    if "jr" in st.session_state:
        return
    df, err = pd.DataFrame(columns=journal.COLS), None
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            df = store.read("판단기록", journal.COLS)
        except Exception as e:
            err = str(e)  # 읽지 못한 채로 저장하면 기존 기록을 덮어쓰므로 저장을 막는다
    st.session_state.jr, st.session_state.jr_err = journal.clean(df), err


def save_journal(new_rows):
    """new_rows: 기록 dict 목록. 반환: (성공, 문구)"""
    store = get_store()
    if store is not None and (st.session_state.get("load_error") or st.session_state.get("jr_err")):
        return False, "구글 시트에서 판단 기록을 읽지 못한 상태라 저장을 막았어요."
    cur = st.session_state.jr
    df = journal.clean(pd.concat([cur, pd.DataFrame(new_rows)], ignore_index=True))
    out = df.copy()
    out["날짜"] = out["날짜"].dt.strftime("%Y-%m-%d")
    try:
        if store is not None:
            store.write("판단기록", out)
        st.session_state.jr = df
        return True, f"{len(new_rows)}건을 기록했어요." + ("" if store is not None else "(데모 모드라서 저장되지는 않아요)")
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"


def holding_agg(code):
    """같은 종목을 여러 증권사에서 들고 있으면 수량 합계와 가중 평균단가로 합친다."""
    h = st.session_state.hold
    hs = h[h["종목코드"] == code]
    if not len(hs):
        return None
    q = float(hs["수량"].sum())
    avg = float((hs["수량"] * hs["평균단가"]).sum() / q) if q > 0 else None
    return {"name": hs["종목명"].iloc[0] or code, "tag": hs["꼬리표"].iloc[0], "qty": q, "avg": avg, "since": (lambda s: s.min() if len(s) else None)(hs["매수일"][hs["매수일"] != ""]) if "매수일" in hs else None}


def card_for(code, tag, avg, qty, since=None):
    """종합 판단 카드. 시세가 없으면 None."""
    df = hist_kr(code)
    ind = sg.indicators(df, since or None)
    if ind is None:
        return None
    rules = st.session_state.rules
    lv = sg.structure(ind, tag, rules)
    sres = safety_for(code)
    counts = safety.summarize(sres[1]) if sres else None
    return judge.build_card(ind["price"], avg or None, qty or None, tag, ind, lv, cons_for(code), st.session_state.get("flow_sum", {}).get(code),
                            counts, rules, st.session_state.get("total_val"))


def journal_due():
    """(마지막 기록 후 경과일 또는 None, 점검이 필요한 보유 종목 코드 목록)"""
    jr = st.session_state.get("jr")
    hold = st.session_state.get("hold")
    if hold is None:
        return None, []
    today = pd.Timestamp.today().normalize()
    last = None if jr is None or jr.empty else int((today - jr["날짜"].max()).days)
    due = []
    for code in hold["종목코드"].drop_duplicates():
        g = jr[jr["종목코드"] == code] if jr is not None and len(jr) else None
        if g is None or g.empty or (today - g["날짜"].max()).days > JOURNAL_DUE_DAYS:
            due.append(code)
    return last, due



# ---------- 매매 계획 (구글 시트 '매매계획' 탭) ----------
def load_plans():
    if "plans" in st.session_state:
        return
    df, err = pd.DataFrame(columns=plan.COLS), None
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            df = store.read("매매계획", plan.COLS)
        except Exception as e:
            err = str(e)  # 읽지 못한 채로 저장하면 기존 계획을 덮어쓰므로 저장을 막는다
    st.session_state.plans, st.session_state.plans_err = plan.clean(df), err


def save_plans(df):
    """df: 전체 계획 표. 반환: (성공, 문구)"""
    store = get_store()
    if store is not None and (st.session_state.get("load_error") or st.session_state.get("plans_err")):
        return False, "구글 시트에서 매매 계획을 읽지 못한 상태라 저장을 막았어요."
    clean_df = plan.clean(df)
    out = clean_df.copy()
    out["생성일"] = out["생성일"].dt.strftime("%Y-%m-%d")
    try:
        if store is not None:
            store.write("매매계획", out)
        st.session_state.plans = clean_df
        return True, "저장했어요." + ("" if store is not None else "(데모 모드라서 저장되지는 않아요)")
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"


def sizing_assets():
    """매수 크기 계산에 쓰는 총자산(주식 평가금액 + 마지막으로 기록한 현금). 없으면 None."""
    val = st.session_state.get("total_val") or 0.0
    snaps = perf.clean_snaps(st.session_state.get("perf_snaps", pd.DataFrame(columns=perf.SNAP_COLS)))
    cash = float(snaps["현금"].dropna().iloc[-1]) if len(snaps) and snaps["현금"].notna().any() else 0.0
    return (val + cash) if (val + cash) > 0 else None


def entry_for(card, price, invalid=None):
    return entry.plan_entry(price, card, sizing_assets(), st.session_state.risk, invalid)


def plan_rows(rules):
    """진행 중인 계획의 점검 결과 목록. 각 항목: (계획 dict, 평가 dict)"""
    pl = st.session_state.get("plans")
    out = []
    if pl is None or pl.empty:
        return out
    for p in pl[pl["상태"] == "진행"].to_dict("records"):
        df = hist_kr(p["종목코드"])
        closes = df["Close"][df.index >= p["생성일"]] if len(df) else None
        out.append((p, plan.evaluate(p, closes)))
    return out


# ---------- 데이터 정리 ----------
def clean_hold(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in HOLD_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[HOLD_COLS]
    df["종목코드"] = df["종목코드"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df = df[df["종목코드"].ne("") & df["종목코드"].ne("None") & df["종목코드"].ne("nan")]
    df["종목코드"] = df["종목코드"].str.zfill(6)
    df["수량"] = pd.to_numeric(df["수량"], errors="coerce").fillna(0)
    df["평균단가"] = pd.to_numeric(df["평균단가"], errors="coerce").fillna(0)
    df["증권사"] = df["증권사"].where(df["증권사"].isin(BROKERS), BROKERS[0])
    df["꼬리표"] = df["꼬리표"].where(df["꼬리표"].isin(TAGS), TAGS[0])
    df["종목명"] = df["종목명"].fillna("").astype(str)
    df["매수일"] = df["매수일"].fillna("").astype(str).str.slice(0, 10).replace({"None": "", "nan": "", "NaT": ""})
    return df.reset_index(drop=True)


def clean_kakao(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in KAKAO_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[KAKAO_COLS]
    df["티커"] = df["티커"].astype(str).str.strip().str.upper()
    df = df[df["티커"].ne("") & df["티커"].ne("NONE") & df["티커"].ne("NAN")]
    df["하루금액"] = pd.to_numeric(df["하루금액"], errors="coerce").fillna(0)
    df["횟수"] = pd.to_numeric(df["횟수"], errors="coerce")
    for c in ("기준일", "시작일"):
        df[c] = df[c].fillna("").astype(str).str.slice(0, 10).replace({"None": "", "nan": "", "NaT": ""})
    df["종목명"] = df["종목명"].fillna("").astype(str)
    return df.reset_index(drop=True)


def kakao_backfill_start(df):
    """저장할 때: 횟수를 적었으면 시작일을 역산해서 채우고 횟수·기준일 칸은 비운다.
    횟수도 시작일도 없으면 오늘부터 세기 시작한다. 횟수와 시작일이 둘 다 있으면 횟수가 우선이다."""
    d = df.copy()
    today = pd.Timestamp.today().normalize()
    fx = hist_fx()
    for i in d.index:
        cnt, start = d.at[i, "횟수"], d.at[i, "시작일"]
        if pd.notna(cnt):
            asof = d.at[i, "기준일"] or today.date().isoformat()
            res = market.kakao_value(hist_us(d.at[i, "티커"]), fx, 1.0, None, cnt, asof)
            if res and res["start"]:
                d.at[i, "시작일"] = res["start"]
            else:  # 시세를 못 가져왔거나 아직 안 산 경우: 영업일로 추정한다
                d.at[i, "시작일"] = perf.kakao_start({"횟수": cnt, "기준일": asof, "시작일": ""}).date().isoformat()
        elif not start:
            d.at[i, "시작일"] = today.date().isoformat()
        d.at[i, "횟수"], d.at[i, "기준일"] = np.nan, ""
    return d


def load_tables(force=False):
    if force:
        for k in ("hold", "kakao", "load_error"):
            st.session_state.pop(k, None)
    if "hold" in st.session_state:
        return
    store = get_store()
    if store is None:
        st.session_state.hold = DEMO_HOLD.copy()
        st.session_state.kakao = DEMO_KAKAO.copy()
        st.session_state.load_error = None
        return
    try:
        st.session_state.hold = clean_hold(store.read("보유종목", HOLD_COLS))
        st.session_state.kakao = clean_kakao(store.read("정기매수", KAKAO_COLS))
        st.session_state.load_error = None
    except Exception as e:  # 불러오기에 실패하면 저장을 막아 시트를 덮어쓰지 않게 한다
        st.session_state.hold = pd.DataFrame(columns=HOLD_COLS)
        st.session_state.kakao = pd.DataFrame(columns=KAKAO_COLS)
        st.session_state.load_error = str(e)



# ---------- DART 재무·공시 (구글 시트에 캐시) ----------
DART_CLIENT_VER = "5"  # DartClient 코드를 바꾸면 이 숫자를 올려서 예전 객체가 재사용되지 않게 한다


@st.cache_resource(show_spinner=False)
def _make_dart(key, ver):
    cget, cput = _dart_cache_io()
    return dart_data.DartClient(key, cget, cput)


def _dart_cache_io():
    """DART 회사 목록 보관본(구글 시트 '회사코드' 탭). 반환: (읽기 함수, 쓰기 함수)"""
    store = get_store()
    if store is None:
        return None, None

    def cget():
        df = store.read("회사코드", ["종목코드", "고유번호", "갱신일"])
        if df.empty:
            return None
        return dict(zip(df["종목코드"].astype(str).str.zfill(6), df["고유번호"].astype(str))), str(df["갱신일"].iloc[0])

    def cput(m):
        today = dt.date.today().isoformat()
        store.write("회사코드", pd.DataFrame({"종목코드": list(m), "고유번호": list(m.values()), "갱신일": today}))

    return cget, cput


def get_dart():
    """반환: (DART 클라이언트 또는 None, 안내 문구). 연결에 실패하면 5분 동안은 다시 시도하지 않아서 화면이 느려지지 않는다."""
    import time
    key = secret("DART_API_KEY")
    if not key:
        return None, "DART_API_KEY가 설정되어 있지 않아요. Streamlit Secrets에 DART_API_KEY를 넣어 주세요."
    ts = st.session_state.get("dart_fail_ts")
    if ts and time.time() - ts < 300:
        return None, st.session_state.get("dart_fail_msg")
    try:
        return _make_dart(key, DART_CLIENT_VER), None
    except Exception as e:
        msg = f"DART에 연결하지 못했어요: {dart_data.redact(e)[:160]} 5분 뒤에 자동으로 다시 시도해요."
        st.session_state["dart_fail_ts"], st.session_state["dart_fail_msg"] = time.time(), msg
        return None, msg


def load_fincache():
    if "fincache" in st.session_state:
        return
    cache = {}
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            for r in store.read("재무캐시", CACHE_COLS).to_dict("records"):
                try:
                    cache[str(r["종목코드"]).zfill(6)] = json.loads(r["데이터"])
                except Exception:
                    pass
        except Exception:
            pass
    st.session_state.fincache = cache


def save_fincache():
    store = get_store()
    if store is None or st.session_state.get("load_error"):
        return
    rows = [[c, d.get("fetched", ""), json.dumps(d, ensure_ascii=False)] for c, d in st.session_state.fincache.items()]
    store.write("재무캐시", pd.DataFrame(rows, columns=CACHE_COLS))


def is_fresh(d):
    try:
        age = (dt.date.today() - dt.date.fromisoformat(d["fetched"])).days
    except Exception:
        return False
    return age <= (0 if d.get("errors") else FRESH_DAYS)


def fetch_company(code):
    """한 종목을 DART에서 가져와 캐시에 넣는다. 반환: (데이터, 오류문구)"""
    dart, err = get_dart()
    if dart is None:
        return None, err
    try:
        d = dart_data.fetch_all(dart, code)
    except Exception as e:
        return None, dart_data.redact(e)
    st.session_state.fincache[code] = d
    return d, None


@st.cache_data(ttl=21600, show_spinner=False)
def _admin_cached():
    return sorted(discover.load_admin_codes())


def admin_codes():
    """관리종목 코드 집합. 거래소 목록을 못 불러오면 None(확인 불가). 실패하면 10분 동안은 다시 시도하지 않는다."""
    import time
    if time.time() - st.session_state.get("admin_err_ts", 0) < 600:
        return None
    try:
        s = set(_admin_cached())
        st.session_state.pop("admin_err", None)
        return s
    except Exception as e:
        st.session_state["admin_err"] = str(e)[:300]
        st.session_state["admin_err_ts"] = time.time()
        return None


def admin_notice():
    """관리종목 목록을 못 불러왔을 때 안내 문구(아니면 None)."""
    admin_codes()
    e = st.session_state.get("admin_err")
    return None if not e else f"거래소 관리종목 목록을 불러오지 못해서 '관리종목 지정 없음'은 확인 불가로 표시해요. (사유: {e})"


def safety_for(code):
    """캐시에 있는 DART 데이터로 안전 점검. 아직 안 가져왔으면 None."""
    d = st.session_state.get("fincache", {}).get(code)
    if d is None:
        return None
    df = hist_kr(code)
    price = float(df["Close"].iloc[-1]) if len(df) else None
    adm = admin_codes()
    m = safety.derive(d, price, market.avg_trading_value_eok(df), None if adm is None else (code in adm))
    if m["cap_jo"] is None and not m["is_pref"]:
        hint = st.session_state.get("disc_cap", {}).get(code)  # DART 주식 수를 못 구했으면 종목 목록의 시가총액을 쓴다
        if hint and hint == hint:
            m["cap_jo"] = float(hint)
    return m, safety.evaluate(m, st.session_state.safe)


# ---------- 시세(캐시) ----------
# 화면을 다시 그릴 때마다 이 파일이 처음부터 실행되므로, 시세는 다시 실행돼도 남는 저장소(cache_resource)에 둔다.
# 이 앱은 사용자가 한 명이라 공유해도 안전하고, 스레드에서 동시에 가져와도 안전하다.
@st.cache_resource
def _price_store():
    return {}


_PRICE_CACHE = _price_store()
PRICE_TTL = 900


def _cached(key, fn, ttl=PRICE_TTL):
    now = time.time()
    hit = _PRICE_CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    df = fn()
    # 실패(빈 결과)는 1분만 기억해서 곧 다시 시도한다
    _PRICE_CACHE[key] = (now if len(df) else now - (ttl - 60), df)
    return df


def hist_kr(code):
    return _cached(("kr", code), lambda: market.history_kr(code))


def hist_us(ticker):
    return _cached(("us", ticker), lambda: market.history_us(ticker))


def hist_index():
    return _cached(("idx",), lambda: market.history_index(), 3600)


def hist_fx():
    return _cached(("fx",), lambda: market.history_fx())


def prefetch_prices():
    """보유·관심 종목과 정기매수 시세를 동시에 가져와 캐시를 채운다. 하나씩 가져오는 것보다 훨씬 빠르다."""
    from concurrent.futures import ThreadPoolExecutor
    now = time.time()
    jobs = []
    plan_codes = list(st.session_state["plans"]["종목코드"]) if "plans" in st.session_state else []
    for c in dict.fromkeys(list(st.session_state.hold["종목코드"]) + list(st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS))["종목코드"]) + plan_codes):
        if not (("kr", c) in _PRICE_CACHE and now - _PRICE_CACHE[("kr", c)][0] < PRICE_TTL):
            jobs.append(lambda c=c: hist_kr(c))
    for t in dict.fromkeys(st.session_state.kakao["티커"]):
        if not (("us", t) in _PRICE_CACHE and now - _PRICE_CACHE[("us", t)][0] < PRICE_TTL):
            jobs.append(lambda t=t: hist_us(t))
    if len(st.session_state.kakao) and not (("fx",) in _PRICE_CACHE and now - _PRICE_CACHE[("fx",)][0] < PRICE_TTL):
        jobs.append(hist_fx)
    if not jobs:
        return
    with st.spinner(f"시세 {len(jobs)}개를 가져오는 중이에요…"):
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda f: f(), jobs))


@st.cache_data(ttl=21600, show_spinner=False)
def listing_cached():
    return discover.load_listing()


@st.cache_data(ttl=86400, show_spinner=False)
def bulk_cached(codes):
    dart, err = get_dart()
    if dart is None:
        raise RuntimeError(err)
    fin, errs = discover.bulk_financials(dart, list(codes))
    if not fin and errs:
        raise RuntimeError(errs[0])  # 실패한 결과는 캐시에 남기지 않는다
    return fin, errs



# ---------- 수급 (네이버 금융, 세션에 30분 보관) ----------
FLOW_TTL = 1800


def get_flow(code, force=False):
    """한 종목의 수급. 반환: (DataFrame, 오류문구). 실패하면 5분 동안은 다시 시도하지 않는다."""
    import time
    store = st.session_state.setdefault("flow_df", {})
    errs = st.session_state.setdefault("flow_err", {})
    now = time.time()
    if not force and code in store and now - store[code][1] < FLOW_TTL:
        return store[code][0], None
    if not force and code in errs and now - errs[code][1] < 300:
        return None, errs[code][0]
    try:
        df = flows.fetch_flow(code)
    except Exception as e:
        errs[code] = (str(e)[:200], now)
        return None, errs[code][0]
    store[code] = (df, now)
    errs.pop(code, None)
    st.session_state.setdefault("flow_sum", {})[code] = flows.summarize(df)
    return df, None


def fetch_flows_bulk(codes, progress=None):
    """여러 종목의 수급을 4개씩 동시에 가져온다. 반환: 오류 목록"""
    import time
    from concurrent.futures import ThreadPoolExecutor
    now = time.time()
    store = st.session_state.setdefault("flow_df", {})
    errs = st.session_state.setdefault("flow_err", {})
    todo = [c for c in dict.fromkeys(codes) if not (c in store and now - store[c][1] < FLOW_TTL)]
    errors = []

    def one(c):
        try:
            return c, flows.fetch_flow(c), None
        except Exception as e:
            return c, None, str(e)[:200]

    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (c, df, e) in enumerate(ex.map(one, todo)):
            if e:
                errs[c] = (e, now)
                errors.append(f"{st.session_state.get('disc_names', {}).get(c, c)}: {e}")
            else:
                store[c] = (df, now)
                errs.pop(c, None)
                st.session_state.setdefault("flow_sum", {})[c] = flows.summarize(df)
            if progress:
                progress(i + 1, len(todo))
    return errors


def flow_txt(fs):
    return "-" if not fs else f"외국인 {fs['f5']:+,.0f}억 · 기관 {fs['i5']:+,.0f}억"



# ---------- 증권사 컨센서스 (네이버, 6시간 보관 + 구글 시트에 일별 기록) ----------
CONS_TTL = 21600


def load_cons_hist():
    if "cons_hist" in st.session_state:
        return
    df, err = pd.DataFrame(columns=CONS_COLS), None
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            df = store.read("컨센서스기록", CONS_COLS)
        except Exception as e:
            err = str(e)  # 읽지 못한 채로 저장하면 기존 기록을 덮어쓰므로 기록을 멈춘다
    st.session_state.cons_hist, st.session_state.cons_hist_err = df, err


def record_cons(items):
    """items: [(종목코드, 데이터)]. 오늘 값이 아직 없을 때만 한 줄씩 추가한다(하루 한 번)."""
    h = st.session_state.get("cons_hist")
    if h is None or st.session_state.get("cons_hist_err"):
        return
    today = dt.date.today().isoformat()
    rows = [[today, c, int(round(d["target"])), d["score"] if d["score"] is not None else ""]
            for c, d in items if d and d.get("has") and d.get("target") and not ((h["종목코드"] == c) & (h["날짜"] == today)).any()]
    if not rows:
        return
    new = pd.concat([h, pd.DataFrame(rows, columns=CONS_COLS)], ignore_index=True).tail(3000)
    store = get_store()
    try:
        if store is not None:
            store.write("컨센서스기록", new)
        st.session_state.cons_hist = new
    except Exception:
        pass  # 기록 실패는 화면 표시에 영향을 주지 않는다


def get_cons(code, force=False):
    import time
    store = st.session_state.setdefault("cons_df", {})
    errs = st.session_state.setdefault("cons_err", {})
    now = time.time()
    if not force and code in store and now - store[code][1] < CONS_TTL:
        return store[code][0], None
    if not force and code in errs and now - errs[code][1] < 300:
        return None, errs[code][0]
    try:
        d = consensus.fetch(code)
    except Exception as e:
        errs[code] = (str(e)[:250], now)
        return None, errs[code][0]
    store[code] = (d, now)
    errs.pop(code, None)
    record_cons([(code, d)])
    return d, None


def fetch_cons_bulk(codes, progress=None):
    import time
    from concurrent.futures import ThreadPoolExecutor
    now = time.time()
    store = st.session_state.setdefault("cons_df", {})
    errs = st.session_state.setdefault("cons_err", {})
    todo = [c for c in dict.fromkeys(codes) if not (c in store and now - store[c][1] < CONS_TTL)]
    errors, got = [], []

    def one(c):
        try:
            return c, consensus.fetch(c), None
        except Exception as e:
            return c, None, str(e)[:250]

    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, (c, d, e) in enumerate(ex.map(one, todo)):
            if e:
                errs[c] = (e, now)
                errors.append(f"{st.session_state.get('disc_names', {}).get(c, c)}: {e}")
            else:
                store[c] = (d, now)
                errs.pop(c, None)
                got.append((c, d))
            if progress:
                progress(i + 1, len(todo))
    record_cons(got)
    return errors


def cons_for(code):
    cd = st.session_state.get("cons_df", {}).get(code)
    return cd[0] if cd else None


def cons_change(code):
    """기록에서 목표가가 며칠 전보다 얼마나 바뀌었는지. 기록이 부족하면 None."""
    h = st.session_state.get("cons_hist")
    if h is None or h.empty:
        return None
    d = h[h["종목코드"] == code].copy()
    d["날짜"] = pd.to_datetime(d["날짜"], errors="coerce")
    d["목표가"] = pd.to_numeric(d["목표가"], errors="coerce")
    d = d.dropna(subset=["날짜", "목표가"]).sort_values("날짜")
    if len(d) < 2:
        return None
    last = d.iloc[-1]
    old = d[d["날짜"] <= last["날짜"] - pd.Timedelta(days=25)]
    base = old.iloc[-1] if len(old) else d.iloc[0]
    days = (last["날짜"] - base["날짜"]).days
    return None if days < 7 else (days, float((last["목표가"] / base["목표가"] - 1) * 100))



# ---------- 종합 점수 ----------
def evidence_text(h):
    hyp = st.session_state.get("hyp")
    if hyp is None or hyp.empty:
        return "미검증"
    g = hyp[hyp["가설"] == h]
    if g.empty:
        return "미검증"
    r = g.iloc[-1]
    return f"{r['등급']} ({r['실행일']})"


def score_for(code, tag, avg, qty, since=None, card=None):
    """종합 점수. 시세가 없으면 None."""
    ind = sg.indicators(hist_kr(code), since or None)
    if ind is None:
        return None
    if card is None:
        card = card_for(code, tag, avg, qty, since)
    sres = safety_for(code)
    m, items = sres if sres else (None, None)
    return score.compute(ind, st.session_state.get("flow_sum", {}).get(code), m, items, cons_for(code), cons_change(code), card,
                         st.session_state.weights, evidence_text)


def render_score(sc, name):
    st.subheader(f"종합 점수 — {name}")
    if sc is None or sc["total"] is None:
        st.info("점수를 계산할 근거가 없어요. 수급·컨센서스·재무를 가져오면 근거가 늘어나요.")
        return
    k = st.columns(4)
    k[0].metric("종합 점수", f"{sc['total']:.0f}점", sc["band"], delta_color="off",
                help="이길 확률이 아니라 근거들이 얼마나 한 방향으로 모이는지를 요약한 숫자예요. 50점이 중립이고 65점 이상이면 우호적인 근거가 모이는 쪽, 35점 미만이면 불리한 근거가 많은 쪽이에요.")
    k[1].metric("반영한 근거", f"{sc['n_avail']}/{sc['n_total']}", ", ".join(sc["missing"]) + " 없음" if sc["missing"] else "모두 있음", delta_color="off",
                help="가격 흐름, 수급, 재무·안전, 컨센서스, 진입 구조 중 몇 가지를 반영했는지예요. 3가지 미만이면 점수는 참고만 하세요.")
    ag = sc["agree"]
    k[2].metric("근거 일치도", "-" if not ag or ag[1] == 0 else f"{ag[0]}/{ag[1]}", help="뚜렷한(±10점 이상) 근거 종류 중 종합 방향과 같은 방향인 개수예요. 낮으면 근거가 엇갈린다는 뜻이에요.")
    k[3].metric("엇갈리는 근거", f"{len(sc['conflicts'])}쌍", help="한쪽은 +25 이상 우호적인데 다른 쪽은 -25 이하로 불리한 근거 종류의 쌍이에요.")
    for cf in sc["conflicts"]:
        st.warning("근거가 엇갈려요: " + cf)
    fam = pd.DataFrame([{"근거": f["name"], "점수": f["score"], "가중치": f["weight"]} for f in sc["families"] if f["score"] is not None])
    if len(fam):
        st.altair_chart(alt.Chart(fam).mark_bar().encode(
            y=alt.Y("근거:N", sort=score.FAMILIES, title=None), x=alt.X("점수:Q", scale=alt.Scale(domain=[-100, 100]), title="불리 ← 0 → 우호"),
            color=alt.condition(alt.datum["점수"] > 0, alt.value("#D93A33"), alt.value("#2A63D4")),
            tooltip=["근거", alt.Tooltip("점수:Q", format="+.0f"), "가중치"]).properties(height=36 * len(fam) + 30), width="stretch")
    rows = [{"근거": f["name"], "항목": i["항목"], "점수": i["점수"], "설명": i["설명"], "검증 결과": i["검증"]} for f in sc["families"] for i in f["items"]]
    with st.expander("항목별 점수와 근거 보기", expanded=False):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={"점수": st.column_config.NumberColumn(format="%+.0f")})
        st.caption("'검증 결과'는 가설 실험실에서 그 항목을 시험한 결과예요. '미검증'은 아직 시험하지 못한 항목이에요.")
    with st.expander("이 점수 읽는 법"):
        st.markdown("- **이길 확률이 아니에요.** 가격 흐름, 수급, 재무·안전, 컨센서스, 진입 구조(손익비)가 **얼마나 한 방향으로 모이는지**를 한 숫자로 요약한 거예요.\n"
                    "- 근거 종류마다 -100(불리)~+100(우호)으로 점수를 매기고, 규칙 탭의 **가중치**로 평균을 내요. 근거가 빠진 종류는 제외하고, 근거가 적을수록 50점 쪽으로 당겨요.\n"
                    "- 중요한 건 숫자보다 **근거가 같은 방향인지(일치도)**예요. 한 근거만 매우 좋고 나머지가 나쁘면 점수는 중립에 가깝게 나와요.\n"
                    "- **가중치와 점수 기준은 제가 정한 시작값이고 검증 전이에요.** 서로 겹치는 근거(예: 추세와 3개월 수익률)가 있어서 같은 이야기를 두 번 세는 효과도 있어요.\n"
                    "- 판단 기록과 점수 기록이 쌓이면 **점수대별 실제 성과**를 확인해서 가중치를 고쳐 나가요(종합 순위, 판단 기록 화면).\n"
                    "- 점수가 높다고 사라는 뜻도, 낮다고 팔라는 뜻도 아니에요.")


# ---------- 계산 ----------
def build_positions(hold, rules):
    rows = []
    for r in hold.to_dict("records"):
        df = hist_kr(r["종목코드"])
        ind = sg.indicators(df, r.get("매수일") or None)
        price = float(df["Close"].iloc[-1]) if len(df) else float("nan")
        qty, avg = float(r["수량"]), float(r["평균단가"])
        prot = sg.protect(ind, r["꼬리표"], avg, rules) if ind else None
        lvl = sg.structure(ind, r["꼬리표"], rules) if ind else None
        sup0 = lvl["supports"][0] if lvl and lvl["supports"] else None
        scv = score_for(r["종목코드"], r["꼬리표"], avg, qty, r.get("매수일") or None) if ind else None
        cdat = cons_for(r["종목코드"])
        cups = consensus.upside(cdat["target"], price) if cdat and cdat.get("has") else float("nan")
        fs = st.session_state.get("flow_sum", {}).get(r["종목코드"])
        sigs = sg.signals(ind, r["꼬리표"], avg, rules, fs) if ind and avg > 0 else []
        sres = safety_for(r["종목코드"])
        if sres is None:
            s_label, s_fail = "미조회", 0
        else:
            s_label, s_fail = safety.headline(sres[1]), safety.summarize(sres[1])["fail"]
        verdict = sg.verdict(sigs) if ind else "시세 부족"
        if s_fail and verdict in ("보유 유지", "추가매수 검토"):
            verdict = "지켜보기"
        sig_txt = ", ".join(s["title"] for s in sigs) or "-"
        if s_fail:
            sig_txt = ("" if sig_txt == "-" else sig_txt + ", ") + "안전 기준 미달"
        rows.append(
            {
                "증권사": r["증권사"], "종목명": r["종목명"] or r["종목코드"], "종목코드": r["종목코드"],
                "꼬리표": r["꼬리표"], "수량": qty, "평균단가": avg, "현재가": price,
                "평가금액": qty * price, "투자원금": qty * avg, "수익금액": qty * (price - avg),
                "수익률": (price / avg - 1) * 100 if avg > 0 else float("nan"),
                "고점 대비(%)": prot["overall_dd"] if prot else float("nan"),
                "수익 보호선(원)": prot["line"] if prot and prot["armed"] else float("nan"),
                "종합점수": scv["total"] if scv and scv["total"] is not None else float("nan"),
                "근거": f"{scv['n_avail']}/{scv['n_total']}" if scv else "-",
                "보호선 상태": sg.protect_state(prot),
                "지지선(원)": sup0["price"] if sup0 else float("nan"),
                "지지선까지(%)": sup0["dist"] if sup0 else float("nan"),
                "평소 되돌림(%)": lvl["depth_med"] if lvl else float("nan"),
                "흐름": sg.flow_of(ind)[0] if ind else "시세 부족",
                "수급(5일)": flow_txt(fs),
                "목표가 여력(%)": cups,
                "안전": s_label,
                "판단": verdict,
                "신호": sig_txt,
            }
        )
    return pd.DataFrame(rows)


def build_kakao(kakao):
    rows = []
    fx = hist_fx()
    for r in kakao.to_dict("records"):
        cnt = None if pd.isna(r.get("횟수")) else r["횟수"]
        res = market.kakao_value(hist_us(r["티커"]), fx, float(r["하루금액"]), r.get("시작일") or None, cnt, r.get("기준일") or None)
        name = r["종목명"] or r["티커"]
        if res is None:
            rows.append({"증권사": "카카오", "종목명": name, "투자원금": float("nan"), "평가금액": float("nan"), "회차": 0, "시작일(계산)": "-", "모자람": False})
        else:
            rows.append({"증권사": "카카오", "종목명": name, "투자원금": res["cost"], "평가금액": res["value"], "회차": res["n"],
                         "시작일(계산)": res["start"] or "-", "모자람": res["short"]})
    return pd.DataFrame(rows)



def _sig(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def get_positions(hold, rules):
    """보유 종목 표 계산(비싼 계산)을 입력이 같으면 다시 하지 않는다."""
    ss = st.session_state
    key = _sig(hold.to_json(), json.dumps(rules, sort_keys=True), json.dumps(ss.safe, sort_keys=True), json.dumps(ss.weights, sort_keys=True),
               sorted((k, round((v or {}).get("f5", 0)), round((v or {}).get("i5", 0))) for k, v in ss.get("flow_sum", {}).items()),
               sorted((k, v[1]) for k, v in ss.get("cons_df", {}).items()),
               sorted((k, d.get("fetched")) for k, d in ss.get("fincache", {}).items()),
               sorted(ss.get("disc_cap", {}).items())[:1], int(time.time() // 120))
    c = ss.get("_pos_cache")
    if c and c[0] == key:
        return c[1]
    df = build_positions(hold, rules)
    ss["_pos_cache"] = (key, df)
    return df


def get_kakao(kakao):
    key = _sig(kakao.to_json(), int(time.time() // 120))
    c = st.session_state.get("_kk_cache")
    if c and c[0] == key:
        return c[1]
    df = build_kakao(kakao)
    st.session_state["_kk_cache"] = (key, df)
    return df


def portfolio_value(rules):
    """보유 주식 평가금액 합계(시세를 못 가져온 종목은 뺀다). 없으면 None."""
    hold, kakao = st.session_state.hold, st.session_state.kakao
    val = 0.0
    if len(hold):
        val += float(get_positions(hold, rules)["평가금액"].sum(skipna=True))
    if len(kakao):
        val += float(get_kakao(kakao)["평가금액"].sum(skipna=True))
    return val if val > 0 else None


def man(n):
    return "-" if pd.isna(n) else f"{n / 1e4:,.0f}만원"


def pct(n):
    return "-" if pd.isna(n) else f"{n:+.1f}%"


# ---------- 화면 ----------
def tab_assets(rules):
    hold, kakao = st.session_state.hold, st.session_state.kakao
    pos = get_positions(hold, rules) if len(hold) else pd.DataFrame()
    kk = get_kakao(kakao) if len(kakao) else pd.DataFrame()

    ok_pos = pos[pos["평가금액"].notna()] if len(pos) else pos
    ok_kk = kk[kk["평가금액"].notna()] if len(kk) else kk
    val = (ok_pos["평가금액"].sum() if len(ok_pos) else 0) + (ok_kk["평가금액"].sum() if len(ok_kk) else 0)
    cost = (ok_pos["투자원금"].sum() if len(ok_pos) else 0) + (ok_kk["투자원금"].sum() if len(ok_kk) else 0)

    st.session_state["total_val"] = val if val > 0 else None
    last_d, due = journal_due()
    if len(hold) and due:
        st.info(f"판단 기록이 {JOURNAL_DUE_DAYS}일 넘게 없는 보유 종목이 {len(due)}개 있어요"
                + ("" if last_d is None else f"(마지막 기록은 {last_d}일 전)") + ". **판단 기록 탭의 월간 점검**에서 한 번에 남길 수 있어요.")
    pr_rows = plan_rows(rules)
    hit = [p["종목명"] or p["종목코드"] for p, ev in pr_rows if ev["level"] == "warn"]
    hit_info = [p["종목명"] or p["종목코드"] for p, ev in pr_rows if ev["level"] == "info"]
    if hit:
        st.warning("매매 계획의 기준에 닿은 종목(무효가격·보호선 이탈): **" + ", ".join(hit) + "** — 매매 계획 화면에서 확인하세요.")
    if hit_info:
        st.info("목표에 도달한 계획: **" + ", ".join(hit_info) + "** — 계획대로 정리할지 점검해 보세요.")
    if val == 0 and not len(hold) and not len(kakao):
        st.info("아직 등록한 종목이 없어요. 아래 '보유 종목 편집'에서 종목을 추가해 보세요.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 평가금액", man(val))
    c2.metric("평가손익", man(val - cost), pct((val / cost - 1) * 100) if cost else None)
    c3.metric("투자 원금", man(cost))
    c4.metric("등록 종목", f"{len(hold)}개 + 정기매수 {len(kakao)}개")

    missing = pos[pos["평가금액"].isna()]["종목명"].tolist() if len(pos) else []
    missing += kk[kk["평가금액"].isna()]["종목명"].tolist() if len(kk) else []
    if missing:
        st.warning("시세를 가져오지 못해 합계에서 제외했어요: " + ", ".join(missing))

    if val > 0:
        by_broker = {}
        if len(ok_pos):
            for b, v in ok_pos.groupby("증권사")["평가금액"].sum().items():
                by_broker[b] = by_broker.get(b, 0) + v
        if len(ok_kk):
            by_broker["카카오"] = by_broker.get("카카오", 0) + ok_kk["평가금액"].sum()
        bdf = pd.DataFrame({"증권사": list(by_broker), "평가금액": list(by_broker.values())})
        bdf["비중"] = bdf["평가금액"] / bdf["평가금액"].sum() * 100
        left, right = st.columns([1, 1])
        with left:
            st.subheader("증권사별 비중")
            st.altair_chart(
                alt.Chart(bdf).mark_arc(innerRadius=55).encode(
                    theta="평가금액:Q", color=alt.Color("증권사:N"), tooltip=["증권사", alt.Tooltip("평가금액:Q", format=",.0f"), alt.Tooltip("비중:Q", format=".1f")]
                ).properties(height=240),
                width="stretch",
            )
        with right:
            st.subheader("쏠림 점검")
            if len(ok_pos):
                w = ok_pos.groupby("종목명")["평가금액"].sum() / val * 100
                over = w[w > CONC_LIMIT]
                if len(over):
                    for n, x in over.items():
                        st.error(f"{n} 한 종목이 전체의 {x:.0f}%예요 (기준 {CONC_LIMIT}%)")
                else:
                    st.success(f"한 종목이 전체의 {CONC_LIMIT}%를 넘지 않아요.")
            st.caption("업종 쏠림은 업종 정보를 연결하는 다음 단계에서 추가돼요.")

    if len(pos):
        st.subheader("보유 종목 점검 요약")
        for v in ["매도·비중 축소 검토", "추가매수 검토", "수익실현 검토", "지켜보기", "보유 유지", "시세 부족"]:
            names = pos[pos["판단"] == v]
            if len(names):
                st.markdown(f"**{v}** : " + ", ".join(f"{r.종목명}({r.증권사})" for r in names.itertuples()))
        st.caption("내가 정한 규칙에 해당하는지 정리한 것이에요. 사고팔지는 직접 판단하세요. 판단은 가격 기준 규칙과 안전 점검 결과를 함께 반영해요. 안전 칸이 미조회면 안전 점검 탭에서 재무 데이터를 먼저 가져오세요.")

        st.subheader("보유 종목")
        c1, c2, c3 = st.columns([1, 1, 3])
        if c1.button("보유 종목 수급 가져오기", key="flow_all"):
            bar = st.progress(0.0, text="수급을 가져오는 중…")
            errs = fetch_flows_bulk(pos["종목코드"].tolist(), lambda i, n: bar.progress(i / max(n, 1), text=f"수급 가져오는 중… ({i}/{n})"))
            st.session_state["flow_msg"] = errs
            st.rerun()
        if c2.button("보유 종목 컨센서스 가져오기", key="cons_all"):
            bar = st.progress(0.0, text="증권사 컨센서스를 가져오는 중…")
            errs = fetch_cons_bulk(pos["종목코드"].tolist(), lambda i, n: bar.progress(i / max(n, 1), text=f"컨센서스 가져오는 중… ({i}/{n})"))
            st.session_state["flow_msg"] = [f"컨센서스 {e}" for e in errs]
            st.rerun()
        c3.caption("수급(30분 저장)과 증권사 컨센서스(6시간 저장)는 네이버 증권 데이터를 읽어와요. 서버에서 접속이 막히면 표시되지 않아요. 목표가 여력은 평균 목표가가 현재가보다 몇 % 높은지예요.")
        for e in st.session_state.pop("flow_msg", []) or []:
            st.warning(f"수급을 못 가져왔어요 — {e}")
        show = pos[["증권사", "종목명", "꼬리표", "수량", "평균단가", "현재가", "평가금액", "수익금액", "수익률", "고점 대비(%)", "수익 보호선(원)", "보호선 상태", "지지선(원)", "지지선까지(%)", "평소 되돌림(%)", "흐름", "수급(5일)", "목표가 여력(%)", "종합점수", "근거", "안전", "판단", "신호"]]
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            column_config={
                "수량": st.column_config.NumberColumn(format="%,d"),
                "평균단가": st.column_config.NumberColumn(format="%,d원"),
                "현재가": st.column_config.NumberColumn(format="%,d원"),
                "평가금액": st.column_config.NumberColumn(format="%,d원"),
                "수익금액": st.column_config.NumberColumn(format="%,d원"),
                "수익률": st.column_config.NumberColumn(format="%.1f%%"),
                "고점 대비(%)": st.column_config.NumberColumn(format="%.1f"),
                "수익 보호선(원)": st.column_config.NumberColumn(format="%,d", help="지금 유효한 보호선이에요. 이탈한 적이 있으면 이탈 뒤의 고점 기준으로 다시 잡힌 값이고, 보호선이 꺼져 있으면(수익이 기준 미만) 비어 있어요."),
                "고점 대비(%)": st.column_config.NumberColumn(format="%.1f", help="최근 1년(또는 매수일 이후) 최고 종가 대비예요."),
                "지지선(원)": st.column_config.NumberColumn(format="%,d", help="과거에 밀렸다가 반등한 가격대 중 현재가 바로 아래에 있는 곳이에요. 참고용이에요."),
                "지지선까지(%)": st.column_config.NumberColumn(format="%.1f"),
                "종합점수": st.column_config.NumberColumn(format="%.0f", help="가격 흐름, 수급, 재무·안전, 컨센서스, 진입 구조를 모은 점수예요(50 중립). 이길 확률이 아니라 근거가 한 방향으로 모이는 정도예요. 근거 칸은 반영한 근거 수예요."),
                "평소 되돌림(%)": st.column_config.NumberColumn(format="%.1f", help="이 종목이 과거에 고점에서 보통 이만큼 밀렸다가 반등했어요(중간값)."),
                "목표가 여력(%)": st.column_config.NumberColumn(format="%.0f"),
            },
        )
    if len(kk):
        st.subheader("카카오 소수점 정기매수")
        kk2 = kk.copy()
        kk2["수익금액"] = kk2["평가금액"] - kk2["투자원금"]
        kk2["손익률"] = (kk2["평가금액"] / kk2["투자원금"] - 1) * 100
        kk2 = kk2.rename(columns={"회차": "현재 횟수"})[["증권사", "종목명", "현재 횟수", "시작일(계산)", "투자원금", "평가금액", "수익금액", "손익률"]]
        st.dataframe(
            kk2, width="stretch", hide_index=True,
            column_config={
                "투자원금": st.column_config.NumberColumn(format="%,d원"),
                "평가금액": st.column_config.NumberColumn(format="%,d원"),
                "수익금액": st.column_config.NumberColumn(format="%,d원"),
                "손익률": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        for r in kk.to_dict("records"):
            if r.get("모자람"):
                st.warning(f"{r['종목명']}: 시세 기록이 짧아서 가장 오래된 회차 일부를 계산에서 뺐어요.")
        st.caption("미국 거래일마다 하루 금액만큼 샀다고 보고 그날 주가·환율로 계산한 추정치예요. 현재 횟수는 시작일부터 거래일마다 하나씩 자동으로 늘어요. 실제 횟수와 어긋나면(휴장일·중단 등) 보유 종목 편집에서 지금 횟수를 다시 적고 저장하면 시작일이 다시 맞춰져요.")

    st.divider()
    with st.expander("보유 종목 편집 (추가·수정·삭제)", expanded=not len(hold)):
        edit_tables()


def edit_tables():
    store = get_store()
    err = st.session_state.get("load_error")
    if store is None:
        st.warning("데모 모드예요. 여기서 고친 내용은 저장되지 않고 새로고침하면 사라져요. 구글 시트를 연결하는 방법은 README를 보세요.")
    if err:
        st.error(f"구글 시트에서 불러오지 못해서 저장을 막았어요. ({err})")
        if st.button("다시 불러오기"):
            load_tables(force=True)
            st.rerun()
        return
    st.markdown("**국내 보유 종목** — 종목코드는 6자리 숫자예요(예: 삼성전자 005930). 행을 추가하려면 맨 아래 빈 줄에 입력하세요.")
    ed = st.data_editor(
        st.session_state.hold, num_rows="dynamic", width="stretch", hide_index=True, key="hold_editor",
        column_config={
            "증권사": st.column_config.SelectboxColumn(options=BROKERS, required=True),
            "꼬리표": st.column_config.SelectboxColumn(options=TAGS, required=True),
            "수량": st.column_config.NumberColumn(min_value=0, step=1),
            "평균단가": st.column_config.NumberColumn(min_value=0, step=1),
            "매수일": st.column_config.TextColumn(help="선택 사항이에요. 2026-03-02 형식. 적어두면 수익 보호선의 고점을 그날 이후로 계산해요."),
        },
    )
    st.markdown("**카카오 소수점 정기매수** — 티커는 미국 종목 약어예요(예: NVDA). **하루 금액**과 **지금까지 산 횟수**(오늘 포함)를 적고 저장하면, 앱이 **시작일을 역산해서 채워 넣고** 횟수 칸은 비워요. "
                "그 뒤로는 미국 거래일마다 현재 횟수가 **자동으로 하나씩 늘어요**(내 자산 탭에서 확인). 횟수를 모르면 시작일만 적어도 되고, 둘 다 비우면 오늘부터 세요. "
                "하루 금액이 바뀌었다면 줄을 하나 더 추가하세요. 실제 횟수와 어긋나면 지금 횟수를 다시 적고 저장하면 시작일이 다시 맞춰져요.")
    ek = st.data_editor(
        st.session_state.kakao, num_rows="dynamic", width="stretch", hide_index=True, key="kakao_editor",
        column_config={
            "하루금액": st.column_config.NumberColumn(min_value=0, step=100),
            "횟수": st.column_config.NumberColumn(min_value=0, step=1, help="지금까지 산 횟수(오늘 포함). 적고 저장하면 시작일을 역산해서 채워요. 저장 뒤에는 비워져요."),
            "기준일": st.column_config.TextColumn(help="선택 사항이에요. 횟수를 센 날짜(2026-10-06 형식). 비워두면 저장한 날이에요."),
            "시작일": st.column_config.TextColumn(help="횟수에서 자동으로 채워져요. 횟수를 모르면 직접 적어도 돼요(2026-03-02 형식)."),
        },
    )
    if st.button("저장", type="primary"):
        h, k = clean_hold(ed), kakao_backfill_start(clean_kakao(ek))
        try:
            if store is not None:
                store.write("보유종목", h)
                store.write("정기매수", k)
            st.session_state.hold, st.session_state.kakao = h, k
            st.session_state.pop("hold_editor", None)
            st.session_state.pop("kakao_editor", None)
            st.success("저장했어요." if store is not None else "임시로 반영했어요(데모 모드).")
            st.rerun()
        except Exception as e:
            st.error(f"저장에 실패했어요: {e}")



def eok(x):
    return "-" if x is None else f"{x / 1e8:,.0f}억"


def render_safety(items):
    for i in items:
        st.markdown(f"{STATUS_ICON[i['status']]} **{i['label']}** — {i['detail']}")


def report_financials(code):
    st.subheader("재무와 안전 점검 (DART)")
    dart, err = get_dart()
    if dart is None:
        st.info(err)
        return
    d = st.session_state.fincache.get(code)
    c1, c2 = st.columns([1, 3])
    label = "재무·공시 가져오기" if d is None else "새로 가져오기"
    if c1.button(label, key=f"fetch_{code}"):
        with st.spinner("DART에서 가져오는 중이에요(10~30초)…"):
            nd, e = fetch_company(code)
        if e:
            st.error(f"가져오지 못했어요: {e}")
        else:
            save_fincache()
            st.rerun()
    if d is None:
        c2.caption("아직 이 종목의 재무 데이터를 가져오지 않았어요.")
        return
    c2.caption(f"갱신일 {d.get('fetched', '-')} · 같은 데이터를 {FRESH_DAYS}일 동안 다시 쓰고, 그 뒤에는 새로 가져오기를 눌러 갱신해요.")
    sres = safety_for(code)
    m, items = sres
    cnt = safety.summarize(items)
    if m.get("is_pref"):
        st.info(f"우선주예요. 재무와 공시는 보통주 회사({m['resolved_code']}) 기준으로 점검하고, 시가총액은 계산하지 않아요.")
    if m.get("name"):
        st.write(f"**{m['name']}**" + (" · 금융업" if m["is_fin"] else "") + (f" · {m['fin_year']}년 사업보고서({'연결' if m['fs'] == 'CFS' else '별도'}) 기준" if m["fin_year"] else ""))
    k = st.columns(5)
    k[0].metric("영업이익(최근)", eok(m["op3"][2]) if m["op3"] and m["op3"][2] is not None else "-")
    k[1].metric("부채비율", "-" if m["debt"] is None else f"{m['debt']:,.0f}%")
    k[2].metric("ROE", "-" if m["roe"] is None else f"{m['roe']:.1f}%")
    k[3].metric("이자보상배율", "-" if m["cover"] is None else ("이자 없음" if m["cover"] >= 999 else f"{m['cover']:.1f}배"))
    k[4].metric("영업현금흐름", eok(m["ocf"]))
    if m["op3"] and any(x is not None for x in m["op3"]) and m["fin_year"]:
        y = m["fin_year"]
        bdf = pd.DataFrame({"연도": [str(y - 2), str(y - 1), str(y)], "영업이익(억원)": [None if x is None else x / 1e8 for x in m["op3"]]}).dropna()
        st.altair_chart(
            alt.Chart(bdf).mark_bar().encode(
                x=alt.X("연도:N", title=None), y=alt.Y("영업이익(억원):Q", title=None),
                color=alt.condition(alt.datum["영업이익(억원)"] < 0, alt.value("#2A63D4"), alt.value("#0F6B63")),
            ).properties(height=180),
            width="stretch",
        )
    if cnt["fail"]:
        st.error(f"안전 기준 미달 {cnt['fail']}개 · 확인 불가 {cnt['unknown']}개 · 통과 {cnt['pass']}개")
    elif cnt["unknown"]:
        st.warning(f"미달은 없지만 확인하지 못한 항목이 {cnt['unknown']}개 있어요.")
    else:
        st.success("내 안전 기준을 모두 통과했어요.")
    render_safety(items)
    st.caption("안전 기준을 통과해도 안전하다는 보증은 아니에요. 관리종목은 거래소 목록으로 확인하고, 투자경고·투자주의 지정은 점검에 포함되지 않아요.")
    with st.expander("근거 데이터 보기 (증권사 앱·네이버 금융과 비교해 보세요)"):
        st.json(d)





def record_form(card, code, name, key, default_decision="보유 유지", score_val=None):
    """판단을 기록한다. 기록하는 순간의 카드 숫자(상승 여력, 하락 위험, 손익비, 등급, 필요 승률)가 함께 저장돼요."""
    c1, c2, c3 = st.columns(3)
    dec = c1.selectbox("결정", journal.DECISIONS, index=journal.DECISIONS.index(default_decision), key=f"{key}_dec")
    reason = c2.selectbox("가장 큰 이유", journal.REASONS, key=f"{key}_why")
    conf = c3.slider("확신도", 1, 5, 3, key=f"{key}_conf", help="1 낮음 ~ 5 높음. 나중에 확신도가 높을 때 실제로 더 잘 맞았는지 확인해요.")
    d1, d2 = st.columns([1, 2])
    inv = d1.number_input("무효가격(손실 한도)", min_value=0.0, value=float(round(card["invalid"])) if card.get("invalid") else 0.0, step=100.0, key=f"{key}_inv",
                          help="이 가격 아래로 내려가면 내 판단이 틀렸다고 보는 가격이에요. 0이면 정하지 않은 거예요.")
    memo = d2.text_input("한 줄 메모", key=f"{key}_memo", placeholder="왜 이 결정을 했는지 한 줄로")
    if st.button("이 판단 기록하기", key=f"{key}_save", type="primary"):
        ok, msg = save_journal([make_record(card, code, name, dec, reason, conf, inv, memo, score_val)])
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()


def make_record(card, code, name, decision, reason, conf, invalid, memo, score_val=None):
    base = card["base"]
    return {"id": dt.datetime.now().strftime("%Y%m%d%H%M%S") + code, "날짜": dt.date.today().isoformat(), "종목코드": code, "종목명": name,
            "결정": decision, "이유": reason, "확신도": conf, "기록가": card["price"], "평단": card["avg"], "수량": card["qty"], "꼬리표": card["tag"],
            "상승여력": card["up"], "하락위험": base["pct"] if base else None, "손익비": card["ratio"], "등급": card["grade"],
            "필요승률": card["need_win"], "무효가격": invalid or None, "메모": memo, "종합점수": score_val}



def render_entry(card, price, code, name, held, key):
    st.subheader("추가로 산다면: 진입 평가" if held else "새로 산다면: 진입 평가")
    st.caption("살 때의 구조를 정리해요. '잃지 않는 종목'을 찾는 게 아니라 **틀렸을 때 잃는 크기를 미리 정하는** 도구예요.")
    e = entry_for(card, price)
    default_inv = float(round(e["invalid"])) if e.get("invalid") else 0.0
    inv_in = st.number_input("무효가격(손실 한도) — 바꾸면 아래가 다시 계산돼요", min_value=0.0, value=default_inv, step=100.0, key=f"{key}_einv",
                             help="이 가격 아래로 내려가면 내 판단이 틀렸다고 보는 가격이에요. 기본값은 가까운 지지선보다 2% 아래예요.")
    if inv_in and inv_in != default_inv:
        e = entry_for(card, price, inv_in)
    k = st.columns(5)
    k[0].metric("진입 판정", e["verdict"].split("(")[0])
    k[1].metric("진입 손익비", "-" if e["ratio"] is None else f"{e['ratio']:.2f}", help="보수적 상승 여력 ÷ 무효가격까지의 하락폭이에요. 기준(규칙 탭, 기본 2.0) 이상이면 유리로 봐요.")
    k[2].metric("틀렸을 때 손실폭", "-" if e["stop_pct"] is None else f"-{e['stop_pct']:.1f}%", help="현재가에서 무효가격까지의 하락폭이에요.")
    k[3].metric("살 수 있는 최대", "-" if e["shares"] is None else f"{e['shares']:,}주", None if not e["amount"] else f"약 {e['amount'] / 1e4:,.0f}만원 · 총자산의 {e['pct_assets']:.1f}%", delta_color="off",
                help="틀렸을 때 총자산의 손실 한도(규칙 탭, 기본 1%)만 잃도록 계산한 최대 수량이에요. 한 종목 최대 비중도 넘지 않아요.")
    k[4].metric("기다릴 가격", "-" if not e["wait_price"] else f"{e['wait_price']:,.0f}원", None if not e["wait_price"] else f"{e['wait_pct']:.1f}%", delta_color="off",
                help="손익비가 기준에 닿는 가격이에요(같은 무효가격과 목표를 가정). 비어 있으면 이미 기준을 넘었거나 계산할 수 없어요.")
    st.markdown(entry.entry_text(e))
    if sizing_assets() is None:
        st.warning("총자산을 알 수 없어서 살 수 있는 크기를 계산하지 못했어요(보유 종목 입력 후, 목표·성과 탭에서 현금을 기록하면 계산돼요).")
    cons = cons_for(code)
    t2 = (cons or {}).get("target") if (cons and cons.get("has")) else None
    if st.button("이 조건으로 매매 계획 만들기", key=f"{key}_mkplan"):
        pl = st.session_state.plans
        if len(pl[(pl["종목코드"] == code) & (pl["상태"] == "진행")]):
            st.warning("이 종목은 이미 진행 중인 계획이 있어요. 매매 계획 화면에서 확인하세요.")
        elif not e.get("invalid"):
            st.error("무효가격을 먼저 정해 주세요.")
        else:
            tag = card["tag"]
            row = {"id": dt.datetime.now().strftime("%Y%m%d%H%M%S") + code, "종목코드": code, "종목명": name, "생성일": dt.date.today().isoformat(), "상태": "진행", "꼬리표": tag,
                   "기준가": price, "수량": e["shares"] or 0, "무효가격": e["invalid"], "목표1": e["up_price"] or price * 1.15,
                   "비율1": 30, "목표2": t2 or (card["ups"][0]["price"] if card["ups"] else price * 1.3), "비율2": 30,
                   "보호폭": st.session_state.rules.get(tag, {}).get("보호폭", 15), "완료1": "", "완료2": "", "메모": ""}
            ok, msg = save_plans(pd.concat([pl, pd.DataFrame([row])], ignore_index=True))
            (st.success if ok else st.error)(msg + (" 매매 계획 화면에서 목표와 정리 비율을 다듬을 수 있어요." if ok else ""))


def render_card(card, name):
    g = card["grade"]
    st.subheader(f"종합 판단 카드 — {name}")
    k = st.columns(5)
    k[0].metric("보수적 상승 여력", "-" if card["up"] is None else f"+{card['up']:.1f}%",
                help="앞으로 오를 수 있는 폭을 일부러 작게 잡은 값이에요. 컨센서스 목표가 여력의 절반과 가까운 저항선 중 작은 값을 써요.")
    k[1].metric("하락 위험(중기)", "-" if not card["base"] else f"{card['base']['pct']:.1f}%",
                help="내려갈 수 있는 폭을 넉넉하게 잡은 값이에요. 중기 지지선이나 이 종목이 과거 큰 조정에서 밀렸던 수준이에요. 반드시 거기까지 간다는 뜻은 아니에요.")
    k[2].metric("손익비", "-" if card["ratio"] is None else f"{card['ratio']:.2f}", g, delta_color="off",
                help="오를 폭 ÷ 내릴 폭이에요. 1이면 같고, 1보다 작으면 내릴 폭이 더 커요. 1.2 이상 보통, 2 이상 유리로 봐요.")
    k[3].metric("필요 승률", "-" if card["need_win"] is None else f"{card['need_win']:.0f}%",
                help="이 손익비로 같은 판단을 여러 번 반복할 때 본전이 되려면 맞혀야 하는 비율이에요. 계산은 내릴 폭 ÷ (오를 폭 + 내릴 폭). 50% 안팎이면 균형, 60%를 넘으면 불리해요.")
    k[4].metric("근거 충족도", f"{card['n_have']}/4", " · ".join(n for n, v in card["have"].items() if not v) or "모두 있음", delta_color="off",
                help="가격 흐름, 컨센서스, 수급, 안전·재무 중 몇 개를 반영했는지예요. 빠진 근거가 있으면 숫자가 덜 정확해요.")
    rows_p = []
    if card["ratio"] is not None:
        rows_p.append({"기간": "가장 보수적(맨 위 숫자)", "오를 폭(%)": card["up"], "내릴 폭(%)": abs(card["base"]["pct"]), "손익비": card["ratio"], "필요 승률(%)": card["need_win"], "등급": g})
    for nm, key in (("단기: 가까운 저항선 ↔ 가까운 지지선", "pair_short"), ("중기: 컨센서스 절반 ↔ 중기 하락", "pair_mid")):
        pr_ = card.get(key)
        if pr_:
            rows_p.append({"기간": nm, "오를 폭(%)": pr_["up"], "내릴 폭(%)": pr_["down"], "손익비": pr_["ratio"], "필요 승률(%)": pr_["need"], "등급": pr_["grade"]})
    if rows_p:
        st.markdown("**기간별 손익비** — 오를 폭과 내릴 폭의 기간을 맞춰서 본 값이에요")
        st.dataframe(pd.DataFrame(rows_p), width="stretch", hide_index=True, column_config={
            "오를 폭(%)": st.column_config.NumberColumn(format="%+.1f"), "내릴 폭(%)": st.column_config.NumberColumn(format="-%.1f"),
            "손익비": st.column_config.NumberColumn(format="%.2f"), "필요 승률(%)": st.column_config.NumberColumn(format="%.0f")})
    try:
        import explain
        with st.expander("이 카드 읽는 법 — 이번 숫자로 풀어쓰기", expanded=True):
            for title, md in explain.card_blocks(card):
                st.markdown(f"**{title}**")
                st.markdown(md)
                st.divider()
    except ImportError:
        pass
    if card["n_have"] < 3:
        st.warning("근거가 부족해요. 수급·컨센서스·안전 점검을 가져오면 더 정확해져요(내 자산 탭의 가져오기 버튼, 안전 점검 탭).")
    for w in card["warns"]:
        st.warning(w)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**상승 후보** (굵게 표시된 것 중 가장 작은 값을 보수적 여력으로 써요)")
        if card["ups"]:
            st.dataframe(pd.DataFrame([{"근거": ("● " if u["core"] else "") + u["label"], "가격(원)": u["price"], "현재가 대비(%)": u["pct"], "설명": u["note"]} for u in card["ups"]]),
                         width="stretch", hide_index=True, column_config={"가격(원)": st.column_config.NumberColumn(format="%,d"), "현재가 대비(%)": st.column_config.NumberColumn(format="%+.1f")})
        else:
            st.caption("올라갈 근거를 찾지 못했어요.")
    with c2:
        st.markdown("**하락 후보**")
        if card["downs"]:
            st.dataframe(pd.DataFrame([{"근거": d["label"] + (" ●" if card["base"] is d else ""), "구분": d["term"], "가격(원)": d["price"], "현재가 대비(%)": d["pct"], "설명": d["note"]} for d in card["downs"]]),
                         width="stretch", hide_index=True, column_config={"가격(원)": st.column_config.NumberColumn(format="%,d"), "현재가 대비(%)": st.column_config.NumberColumn(format="%+.1f")})
        else:
            st.caption("내려갈 지점을 찾지 못했어요(가격 기록이 짧을 수 있어요).")
    if card["scen"]:
        st.markdown("**내 평단 기준 시나리오**")
        st.dataframe(pd.DataFrame(card["scen"]), width="stretch", hide_index=True, column_config={
            "가격": st.column_config.NumberColumn(format="%,d"), "현재가 대비(%)": st.column_config.NumberColumn(format="%+.1f"),
            "내 평단 대비(%)": st.column_config.NumberColumn(format="%+.1f"), "내 손익 변화(원)": st.column_config.NumberColumn(format="%+,d"),
            "총자산 영향(%)": st.column_config.NumberColumn(format="%+.2f")})
    st.caption("예측이 아니라 지금 가진 근거로 잡은 보수적인 범위예요. 확률은 검증 전이라 모른다고 보고, 대신 '필요 승률'을 보여줘요. 내 판단 기록이 쌓이면 실제 승률과 비교해서 이 카드가 쓸 만한지 확인할 수 있어요. 투자 권유가 아니에요.")


def report_structure(ind, tag, price, df):
    st.subheader("가격 흐름: 평소 되돌림과 지지선")
    lv = sg.structure(ind, tag, st.session_state.rules)
    if lv is None:
        st.info("가격 기록이 짧거나 뚜렷한 오르내림이 없어서 흐름을 정리하지 못했어요.")
        return
    k = st.columns(4)
    k[0].metric("평소 되돌림(중간값)", "-" if lv["depth_med"] != lv["depth_med"] else f"{lv['depth_med']:.1f}%", f"{lv['n_pull']}번 기준", delta_color="off")
    k[1].metric("큰 되돌림(하위 25%)", "-" if lv["depth_p75"] != lv["depth_p75"] else f"{lv['depth_p75']:.1f}%")
    k[2].metric("지금 마지막 고점 대비", "-" if lv["cur_dd"] != lv["cur_dd"] else f"{lv['cur_dd']:.1f}%", lv["last_peak_date"], delta_color="off")
    k[3].metric("밀림으로 보는 기준", f"-{lv['thr'] * 100:.0f}%", "이 종목의 변동성에 맞춤", delta_color="off")
    if lv["depth_med"] == lv["depth_med"] and lv["cur_dd"] == lv["cur_dd"]:
        if lv["cur_dd"] >= lv["depth_med"]:
            st.write(f"지금 낙폭({lv['cur_dd']:.1f}%)은 이 종목이 **평소 밀리던 범위(중간값 {lv['depth_med']:.1f}%) 안**이에요.")
        elif lv["cur_dd"] >= lv["depth_p75"]:
            st.write(f"지금 낙폭({lv['cur_dd']:.1f}%)은 평소 중간값({lv['depth_med']:.1f}%)보다 깊지만 **큰 되돌림 수준({lv['depth_p75']:.1f}%) 안**이에요.")
        else:
            st.write(f"지금 낙폭({lv['cur_dd']:.1f}%)은 과거 되돌림 중 **큰 편({lv['depth_p75']:.1f}%)보다도 깊어요.** 평소와 다른 흐름일 수 있어요.")
    rows = []
    for s in lv["supports"]:
        rows.append({"구분": "지지선", "가격(원)": s["price"], "현재가 대비(%)": s["dist"], "반등 횟수": s["touches"],
                     "이후 반등(%)": s["rebound"], "최근 확인일": s["last_date"]})
    for s in lv["resistances"]:
        rows.append({"구분": "저항선", "가격(원)": s["price"], "현재가 대비(%)": s["dist"], "반등 횟수": np.nan, "이후 반등(%)": np.nan, "최근 확인일": s["last_date"]})
    for b in lv["broken"][:2]:
        rows.append({"구분": f"깨진 지지선({b['days']}거래일 전 이탈)", "가격(원)": b["price"], "현재가 대비(%)": b["dist"], "반등 횟수": b["touches"],
                     "이후 반등(%)": b["rebound"], "최근 확인일": b["last_date"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "가격(원)": st.column_config.NumberColumn(format="%,d"), "현재가 대비(%)": st.column_config.NumberColumn(format="%.1f"),
            "이후 반등(%)": st.column_config.NumberColumn(format="%.0f"), "반등 횟수": st.column_config.NumberColumn(format="%d")})
    closes, dates = ind["all_close"][-300:], ind["all_dates"][-300:]
    cdf = pd.DataFrame({"날짜": pd.to_datetime(dates), "종가": closes})
    lines = [{"가격": s["price"], "구분": "지지선"} for s in lv["supports"][:2]] + [{"가격": s["price"], "구분": "저항선"} for s in lv["resistances"][:1]]
    base = alt.Chart(cdf).mark_line(color="#16202C").encode(x=alt.X("날짜:T", title=None), y=alt.Y("종가:Q", scale=alt.Scale(zero=False), title=None))
    layers = [base]
    if lines:
        ldf = pd.DataFrame(lines)
        layers.append(alt.Chart(ldf).mark_rule(strokeDash=[5, 4]).encode(
            y="가격:Q", color=alt.Color("구분:N", scale=alt.Scale(domain=["지지선", "저항선"], range=["#0F6B63", "#D93A33"]))))
    st.altair_chart(alt.layer(*layers).properties(height=260), width="stretch")
    pbs = [p for p in lv["pullbacks"]][-6:]
    if pbs:
        with st.expander("과거에 밀렸다가 반등한 구간 (최근 6번)"):
            st.dataframe(pd.DataFrame([{"고점일": p["peak_date"], "고점": p["peak"], "저점일": p["trough_date"], "저점": p["trough"], "하락(%)": p["depth"],
                                        "이후 반등(%)": p["rebound"], "반등까지(거래일)": p["rebound_days"], "진행 중": "예" if p["ongoing"] else ""} for p in pbs]),
                         width="stretch", hide_index=True, column_config={
                             "고점": st.column_config.NumberColumn(format="%,d"), "저점": st.column_config.NumberColumn(format="%,d"),
                             "하락(%)": st.column_config.NumberColumn(format="%.1f"), "이후 반등(%)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("종가 기준으로 과거 오르내림을 정리한 참고 자료예요. 같은 가격대에서 다시 반등한다는 보장은 없고, 지지선도 깨질 수 있어요. 지지선 이탈 신호는 매도하라는 뜻이 아니라 점검하라는 뜻이에요.")


def report_flow(code):
    st.subheader("수급 흐름 (외국인·기관·개인)")
    df, err = get_flow(code)
    if err:
        st.warning(f"수급 데이터를 가져오지 못했어요: {err}")
        if st.button("다시 시도", key=f"flow_retry_{code}"):
            get_flow(code, force=True)
            st.rerun()
        return
    sm = flows.summarize(df)
    if sm is None:
        st.info("수급 기록이 너무 적어서 계산하지 못했어요.")
        return
    k = st.columns(4)
    k[0].metric("외국인 5일", f"{sm['f5']:+,.0f}억")
    k[0].caption(f"20일 합계 {sm['f20']:+,.0f}억")
    k[1].metric("기관 5일", f"{sm['i5']:+,.0f}억")
    k[1].caption(f"20일 합계 {sm['i20']:+,.0f}억")
    k[2].metric(("개인(추정)" if sm["indiv_est"] else "개인") + " 5일", f"{sm['p5']:+,.0f}억")
    k[2].caption(f"20일 합계 {sm['p20']:+,.0f}억")
    k[3].metric("외국인 보유율", "-" if sm["hold_pct"] is None else f"{sm['hold_pct']:.2f}%")
    k[3].caption("" if sm["hold_chg20"] is None else f"20일 변화 {sm['hold_chg20']:+.2f}%p")
    cf = flows.chart_frame(df, 20)
    for who in cf["주체"].unique():
        st.caption(f"{who} 일별 순매수(억원, 최근 20거래일) — 빨강은 순매수, 파랑은 순매도")
        st.altair_chart(alt.Chart(cf[cf["주체"] == who]).mark_bar().encode(
            x=alt.X("일:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)), y=alt.Y("순매수(억):Q", title=None),
            color=alt.condition(alt.datum["순매수(억)"] > 0, alt.value("#D93A33"), alt.value("#2A63D4")),
            tooltip=["일:N", alt.Tooltip("순매수(억):Q", format=",.1f")]).properties(height=90), width="stretch")
    st.info(flows.read_text(sm))
    st.caption(f"{sm['last']}까지의 자료예요. 외국인 연속 {abs(sm['f_streak'])}일 {'순매수' if sm['f_streak'] > 0 else '순매도'}, "
               f"기관 연속 {abs(sm['i_streak'])}일 {'순매수' if sm['i_streak'] > 0 else '순매도'}. 금액은 순매매량에 그날 종가를 곱한 추정치이고, "
               + ("개인은 제공되지 않아 기관·외국인의 반대로 계산한 값(기타 법인 등 포함)이에요. " if sm["indiv_est"] else "") + "수급은 참고 자료일 뿐 주가를 보장하지 않아요.")
    if sm.get("sum5") is not None and not sm["indiv_est"]:
        st.caption(f"외국인+기관+개인의 합이 5일 {sm['sum5']:+,.0f}억으로 0이 아닌 것은 기타 법인, 회사의 자사주 매입 등 나머지 주체가 따로 있기 때문이에요(네이버 금융 화면의 값과 같아요).")
    if sm.get("p5_given") is not None and sm["indiv_est"]:
        st.caption(f"참고: 네이버가 따로 준 개인 값은 5일 {sm['p5_given']:+,.0f}억이에요. 그런데 외국인+기관+개인의 합이 5일 {sm['sum5']:+,.0f}억으로 0에서 멀어서(보통 기타 법인 몫만큼만 벌어져요), "
                   "증권사 앱 값과 맞는 것이 확인되기 전까지는 개인을 기관·외국인의 반대로 추정해서 써요.")
    with st.expander("수급 원자료 (최근 40거래일)"):
        raw = flows.with_amounts(df).tail(40).iloc[::-1]
        cols = [c for c in ["날짜", "종가", "거래량", "기관", "외국인", "개인", "기관(억)", "외국인(억)", "개인제공(억)", "3주체합(억)", "외국인보유율"] if c in raw.columns]
        st.caption("증권사 앱과 비교할 때는 '기관', '외국인', '개인' 열(순매매량, 단위 주)을 같은 날짜끼리 보세요.")
        st.dataframe(raw[cols], width="stretch", hide_index=True)



def report_consensus(code, price):
    st.subheader("증권사 컨센서스")
    d, err = get_cons(code)
    if err:
        st.warning(f"컨센서스를 가져오지 못했어요: {err}")
        if st.button("다시 시도", key=f"cons_retry_{code}"):
            get_cons(code, force=True)
            st.rerun()
        return
    if not d["has"]:
        st.info("이 종목은 증권사 컨센서스가 없어요(리포트를 내는 증권사가 없거나 적은 종목일 수 있어요).")
        return
    up = consensus.upside(d["target"], price)
    k = st.columns(4)
    k[0].metric("평균 목표주가", f"{d['target']:,.0f}원" if d["target"] else "-")
    k[1].metric("현재가 대비 여력", "-" if up != up else f"{up:+.1f}%")
    k[2].metric("투자의견 평균", "-" if d["score"] is None else f"{d['score']:.2f}점", consensus.label(d["score"]), delta_color="off")
    k[3].metric("컨센서스 기준일", d["date"] or "-")
    ch = cons_change(code)
    if ch:
        st.write(f"**{ch[0]}일 전 기록 대비 평균 목표가 {ch[1]:+.1f}%** — 목표가가 {'올라가고' if ch[1] > 0 else '내려가고' if ch[1] < 0 else '그대로이고'} 있어요.")
    else:
        st.caption("목표가의 변화는 기록이 쌓이면 보여줘요. 앱에서 컨센서스를 가져올 때마다 하루 한 번 구글 시트(컨센서스기록 탭)에 저장돼요.")
    if up == up and up < 0:
        st.warning("현재가가 평균 목표가보다 높아요. 컨센서스는 늦게 갱신돼서 오른 종목에서는 흔한 일이지만, 증권사가 보는 목표 수준을 넘었다는 뜻이에요.")
    st.caption("평균 목표주가는 증권사들의 전망일 뿐이고 틀리는 경우도 많아요. 증권사 의견은 매수가 대부분이라 점수가 높게 나오는 게 보통이라, 점수 자체보다 목표가가 올라가는지 내려가는지와 의견이 바뀌는지가 더 중요해요. 5점 만점(5 적극매수 ~ 1 적극매도)이고, 네이버 증권 데이터예요.")



REP_TTL = 21600


def get_reports(code, force=False):
    """반환: (목록, 안내문 리스트, 오류문구). 6시간 동안은 다시 가져오지 않는다."""
    import time
    store = st.session_state.setdefault("rep_df", {})
    now = time.time()
    if not force and code in store and now - store[code][2] < REP_TTL:
        return store[code][0], store[code][1], None
    try:
        rows, notes = reports.fetch_reports(code)
    except Exception as e:
        return None, None, str(e)[:250]
    store[code] = (rows, notes, now)
    return rows, notes, None


def report_reports(code, price):
    st.subheader("증권사별 리포트 (최근)")
    cached = code in st.session_state.get("rep_df", {})
    if not cached:
        if not st.button("증권사 리포트 목록 가져오기", key=f"rep_{code}"):
            st.caption("종목마다 10~20초 걸려서 버튼을 눌렀을 때만 가져와요. 증권사·날짜·투자의견·목표가와 제목만 읽고, 리포트 본문은 가져오지 않아요.")
            return
    with st.spinner("리포트 목록을 가져오는 중이에요…"):
        rows, notes, err = get_reports(code)
    if err:
        st.warning(f"리포트 목록을 가져오지 못했어요: {err}")
        return
    sm = reports.summarize(rows, price)
    k = st.columns(4)
    k[0].metric("최근 리포트", f"{sm['n']}건")
    k[1].metric("평균 목표가", "-" if not sm["avg"] else f"{sm['avg']:,.0f}원", None if sm["upside"] is None else f"현재가 대비 {sm['upside']:+.1f}%", delta_color="off")
    k[2].metric("목표가 상향 / 하향", f"{sm['ups']} / {sm['downs']}건")
    k[3].metric("매수 계열 의견", "-" if not sm["n_op"] else f"{sm['buy']}/{sm['n_op']}건")
    df = pd.DataFrame([{
        "날짜": r["date"], "증권사": r["firm"], "의견": r["opinion"] or "-", "목표가(원)": r["target"],
        "직전 대비(%)": ((r["target"] / r["prev"] - 1) * 100) if r["target"] and r["prev"] else np.nan,
        "현재가 대비(%)": ((r["target"] / price - 1) * 100) if r["target"] and price else np.nan,
        "제목": r["title"], "원문": r["url"]} for r in rows])
    st.dataframe(df, width="stretch", hide_index=True, column_config={
        "목표가(원)": st.column_config.NumberColumn(format="%,d"), "직전 대비(%)": st.column_config.NumberColumn(format="%.1f"),
        "현재가 대비(%)": st.column_config.NumberColumn(format="%.1f"),
        "원문": st.column_config.LinkColumn("원문", display_text="열기")})
    for n in notes:
        st.warning(n)
    if st.button("새로 가져오기", key=f"rep_re_{code}"):
        get_reports(code, force=True)
        st.rerun()
    st.caption("'직전 대비'는 이 목록 안에서 같은 증권사의 바로 앞 리포트와 비교한 값이라, 목록에 이전 리포트가 없으면 비어 있어요. 증권사 의견은 매수가 대부분이라 개수보다 목표가의 방향과 의견이 바뀌는 리포트가 중요해요. 목표가는 증권사의 전망일 뿐이에요. '원문'은 네이버 금융의 리포트 페이지로 연결돼요.")


def tab_report(rules):
    hold = st.session_state.hold
    opts = {f"{r['종목명'] or r['종목코드']} ({r['종목코드']}) · {r['증권사']}": r for r in hold.to_dict("records")}
    held_codes = set(hold["종목코드"])
    for w in st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS)).to_dict("records"):
        if w["종목코드"] not in held_codes:
            opts[f"{w['종목명'] or w['종목코드']} ({w['종목코드']}) · 관심"] = {
                "증권사": "-", "종목코드": w["종목코드"], "종목명": w["종목명"], "꼬리표": "중장기", "수량": 0, "평균단가": 0}
    labels = list(opts) + ["직접 입력"]
    if st.session_state.get("rep_select") not in labels:
        st.session_state.pop("rep_select", None)
    pick = st.selectbox("종목 선택", labels, key="rep_select")
    if pick == "직접 입력":
        code = st.text_input("종목코드(6자리)", value="005930").strip().zfill(6)
        held, tag_default, avg = None, "중장기", None
    else:
        held = opts[pick]
        code, tag_default, avg = held["종목코드"], held["꼬리표"], float(held["평균단가"]) or None
    tag = st.radio("보는 관점", TAGS, index=TAGS.index(tag_default), horizontal=True)

    df = hist_kr(code)
    ind = sg.indicators(df, (held or {}).get("매수일") or None)
    if len(df) == 0:
        st.error("시세를 가져오지 못했어요. 종목코드를 확인하거나 잠시 뒤 다시 시도해 보세요.")
        return
    if ind is None:
        st.warning("시세 데이터가 너무 적어서(61거래일 미만) 지표를 계산하지 못했어요.")
        return

    flow, flow_desc = sg.flow_of(ind)
    m = st.columns(5)
    m[0].metric("현재가", f"{ind['price']:,.0f}원", pct(ind["day"]))
    m[1].metric("1개월", pct(ind["m1"]))
    m[2].metric("3개월", pct(ind["m3"]))
    m[3].metric("RSI", f"{ind['rsi']:.0f}")
    m[4].metric("52주 위치", f"{ind['pos']:.0f}%")
    st.markdown(f"**{flow}** — {flow_desc}")

    c = df["Close"]
    d = pd.DataFrame({"날짜": df.index, "종가": c.values, "20일선": c.rolling(20).mean().values, "60일선": c.rolling(60).mean().values}).tail(120)
    long = d.melt("날짜", var_name="구분", value_name="가격").dropna()
    st.altair_chart(
        alt.Chart(long).mark_line().encode(
            x=alt.X("날짜:T", title=None),
            y=alt.Y("가격:Q", scale=alt.Scale(zero=False), title=None),
            color=alt.Color("구분:N", scale=alt.Scale(domain=["종가", "20일선", "60일선"], range=["#16202C", "#0F6B63", "#9A5507"])),
            strokeDash=alt.StrokeDash("구분:N", scale=alt.Scale(domain=["종가", "20일선", "60일선"], range=[[1, 0], [1, 0], [5, 4]]), legend=None),
        ).properties(height=320),
        width="stretch",
    )
    st.caption("최근 120거래일 종가와 이동평균선")

    qty_ = float(held["수량"]) if held else 0.0
    card = card_for(code, tag, avg, qty_, (held or {}).get("매수일") or None)
    sc = score_for(code, tag, avg, qty_, (held or {}).get("매수일") or None, card=card)
    render_score(sc, (held or {}).get("종목명") or code)
    st.subheader(f"내 기준으로 보면 ({tag})")
    sigs = sg.signals(ind, tag, avg, rules)
    prot = None
    if avg:
        st.write(f"내 평단 {avg:,.0f}원 · 수익률 {pct((ind['price'] / avg - 1) * 100)}")
        prot = sg.protect(ind, tag, avg, rules)
        if prot:
            if prot["armed"]:
                st.write(f"수익 보호선: **{prot['line']:,.0f}원** — 기준 고점 {prot['peak']:,.0f}원({prot['peak_date']}) − {prot['width']:.0f}% · 지금은 그 고점 대비 {prot['dd']:.1f}% ({sg.protect_state(prot)})")
            else:
                st.caption(f"수익 보호선은 지금 꺼져 있어요. 기준 고점 {prot['peak']:,.0f}원의 수익률이 +{prot['peak_ret']:.0f}%로 시작 기준(+{prot['act']:.0f}%)에 못 미쳐요. ({sg.protect_state(prot)})")
            if prot["last"]:
                l = prot["last"]
                st.caption(f"지난번 이탈: {l['date']}에 보호선 {l['line']:,.0f}원(고점 {l['peak']:,.0f}원 기준)을 {l['price']:,.0f}원으로 이탈했고, 그 가격에서 보호선을 새로 잡았어요.")
    if sigs:
        for s in sigs:
            icon = {"주의": "⚠️", "매수검토": "🟢", "알림": "🔔"}[s["kind"]]
            st.markdown(f"{icon} **{s['title']}** — {s['detail']}")
        st.markdown(f"정리: **{sg.verdict(sigs)}**")
    try:
        import explain
        if sigs or prot:
            with st.expander("신호와 보호선 읽는 법 — 이번 숫자로 풀어쓰기"):
                if prot:
                    st.markdown(explain.protect_text(prot, ind["price"]))
                for s in sigs:
                    hlp = explain.SIGNAL_HELP.get(s["title"])
                    if hlp:
                        st.markdown(f"- **{s['title']}**: {hlp}{evidence_for(s['title'])}")
                if sigs:
                    v = sg.verdict(sigs)
                    st.markdown(f"- **정리: {v}**: {explain.VERDICT_HELP.get(v, '')} (신호가 걸렸다는 뜻이지 사고팔라는 뜻은 아니에요.)")
    except ImportError:
        pass
    else:
        st.write("지금은 내 기준에 걸리는 신호가 없어요.")
    if card:
        render_card(card, (held or {}).get("종목명") or code)
        with st.expander("이 카드로 판단 기록하기"):
            record_form(card, code, (held or {}).get("종목명") or code, f"rf_{code}", "보유 유지" if held else "관망(사지 않음)", sc["total"] if sc else None)
        render_entry(card, ind["price"], code, (held or {}).get("종목명") or code, bool(held and float(held["수량"]) > 0), f"en_{code}")
    report_structure(ind, tag, ind["price"], df)
    report_financials(code)
    report_flow(code)
    report_consensus(code, ind["price"])
    report_reports(code, ind["price"])




# ---------- 종목 발굴 ----------
def disc_filters_ui(f):
    def row(key, label, unit, step=1.0):
        a, b = st.columns([3, 2])
        f[key]["on"] = a.checkbox(label, value=f[key]["on"], key=f"d_on_{key}")
        if unit:
            f[key]["v"] = b.number_input(unit, value=float(f[key]["v"]), step=step, key=f"d_v_{key}")
    p1, p2, _ = st.columns([1, 1, 3])
    if p1.button("가치 중심 기본", key="d_p1"):
        st.session_state.disc_f = copy.deepcopy(discover.DEFAULT_FILTERS)
        for k in list(st.session_state.keys()):
            if k.startswith(("d_on_", "d_v_")) or k == "d_maxn":
                del st.session_state[k]
        st.rerun()
    if p2.button("성장 중심", key="d_p2"):
        g = copy.deepcopy(discover.DEFAULT_FILTERS)
        g["roe"]["v"], g["per"]["v"], g["debt"]["v"], g["growth"]["on"] = 12.0, 25.0, 150.0, True
        st.session_state.disc_f = g
        for k in list(st.session_state.keys()):
            if k.startswith(("d_on_", "d_v_")) or k == "d_maxn":
                del st.session_state[k]
        st.rerun()
    st.markdown("**1단계 — 시장 목록 필터** (바꾸면 아래 '시장 데이터 불러오기'를 다시 눌러요)")
    row("cap", "시가총액", "조원 이상", 0.5)
    row("tv", "일 거래대금", "억원 이상", 10.0)
    f["maxn"] = int(st.number_input("2단계에서 조회할 최대 종목 수(시가총액 큰 순)", min_value=50, max_value=1000, value=int(f["maxn"]), step=50, key="d_maxn"))
    st.markdown("**2단계 — 재무 조건** (바로 반영돼요)")
    row("profit", "최근 영업이익 흑자", None)
    row("loss", "최근 3년 영업적자", "번 이하")
    row("debt", "부채비율", "% 이하", 10.0)
    row("roe", "ROE", "% 이상", 1.0)
    row("per", "PER", "배 이하", 1.0)
    row("pbr", "PBR", "배 이하", 0.1)
    row("growth", "영업이익이 전년보다 증가", None)
    st.caption("PER·PBR은 오늘 시가총액을 최근 사업보고서의 순이익·자본총계로 나눈 추정치예요(지배주주 몫과 올해 실적은 반영되지 않아요). 금융업으로 보이는 종목(이름에 금융·은행·증권·보험 등)은 부채비율 조건을 면제해요.")


def run_discovery(f, manual_codes=None):
    bar = st.progress(0.0, text="종목 목록을 가져오는 중…")
    try:
        uni = discover.manual_listing(manual_codes) if manual_codes else listing_cached()
    except Exception as e:
        bar.empty()
        st.session_state.disc_res = {"fail": str(e)}
        return
    st.session_state.disc_cap = {c: m / 1e12 for c, m in zip(uni["Code"], uni["Marcap"]) if m == m}
    held = set(st.session_state.hold["종목코드"]) if st.session_state.get("disc_excl_held", True) else set()
    s1, notes = discover.stage1(uni, f, exclude=held)
    bar.progress(0.3, text=f"DART 재무 일괄 조회 중… ({len(s1)}개)")
    try:
        fin, errs = bulk_cached(tuple(s1["Code"]))
    except Exception as e:
        bar.empty()
        st.session_state.disc_res = {"fail": f"DART 재무 조회에 실패했어요: {e}", "uni_n": len(uni), "s1_n": len(s1)}
        return
    tbl = discover.metrics_table(s1, fin)
    st.session_state.disc_res = {"uni_n": len(uni), "s1_n": len(s1), "fin_n": len(fin), "tbl": tbl, "notes": notes,
                                 "errors": errs, "manual": bool(manual_codes), "ts": dt.datetime.now().strftime("%m-%d %H:%M"),
                                 "dept": uni["Dept"].replace("", "(비어 있음)").value_counts().head(15).to_dict() if "Dept" in uni else {}}
    bar.progress(1.0, text="완료")


def run_detail(codes):
    bar, errors = st.progress(0.0), []
    for i, c in enumerate(codes):
        name = st.session_state.disc_names.get(c, c)
        bar.progress(i / max(len(codes), 1), text=f"{name} 상세 점검 중… ({i + 1}/{len(codes)})")
        d = st.session_state.fincache.get(c)
        if not (d and is_fresh(d)):
            _, e = fetch_company(c)
            if e:
                errors.append(f"{name}: {e}")
                if any(k in e for k in ("010", "011", "020")):
                    break
    bar.progress(1.0)
    save_fincache()
    fe = fetch_flows_bulk(codes)
    if fe:
        st.warning(f"수급을 가져오지 못한 종목이 {len(fe)}개 있어요(예: {fe[0]})")
    ce = fetch_cons_bulk(codes)
    if ce:
        st.warning(f"컨센서스를 가져오지 못한 종목이 {len(ce)}개 있어요(예: {ce[0]})")
    st.session_state.disc_detail = list(codes)
    if errors:
        st.error("가져오지 못한 종목이 있어요:\n\n" + "\n\n".join(errors))



def _entry_cells(code, ind):
    """종목 발굴 표용: 진입 판정, 손익비, 무효가격, 최대 매수 수량, 기다릴 가격."""
    try:
        if not ind:
            return ("-", np.nan, np.nan, np.nan, np.nan)
        card = card_for(code, "중장기", None, None, None)
        if not card:
            return ("-", np.nan, np.nan, np.nan, np.nan)
        e = entry_for(card, ind["price"])
        return (e["verdict"].split("(")[0], e["ratio"] if e["ratio"] is not None else np.nan, e["invalid"] or np.nan,
                e["shares"] if e["shares"] else np.nan, e["wait_price"] or np.nan)
    except Exception:
        return ("-", np.nan, np.nan, np.nan, np.nan)


def tab_discover():
    st.markdown("시장 전체에서 후보를 좁히는 깔때기예요. **① 시가총액·거래대금 → ② DART 재무 → ③ 남은 상위 후보만 안전 기준·가격 흐름까지 상세 점검**해요. 통과했다고 사라는 뜻은 아니고, 더 살펴볼 후보라는 뜻이에요.")
    dart, err = get_dart()
    if dart is None:
        st.warning(err)
        return
    f = st.session_state.disc_f
    with st.expander("발굴 조건", expanded=False):
        disc_filters_ui(f)
        a, _ = st.columns([1, 3])
        if a.button("조건 저장", key="d_save"):
            ok, msg = save_settings()
            (st.success if ok else st.error)(msg)
    st.session_state.setdefault("disc_excl_held", True)
    st.checkbox("이미 보유한 종목은 제외", key="disc_excl_held")
    res = st.session_state.get("disc_res")
    manual_codes = None
    if res and res.get("fail") and "종목 목록" in res["fail"]:
        st.error(res["fail"])
        st.info("종목 목록 출처에 접속하지 못했어요. 대신 살펴볼 종목코드를 직접 붙여넣어 재무 조건만 적용할 수 있어요(시가총액이 없어서 PER·PBR·시가총액 조건은 빠져요).")
        txt = st.text_area("종목코드 목록 (쉼표나 줄바꿈으로 구분)", key="disc_manual", placeholder="005930, 000660, 035420")
        manual_codes = [x for x in txt.replace("\n", ",").replace(" ", ",").split(",") if x.strip()] or None
    if st.button("시장 데이터 불러오기", type="primary"):
        with st.spinner("시장 데이터를 불러오는 중이에요(1~3분 걸릴 수 있어요)…"):
            run_discovery(f, manual_codes)
        res = st.session_state.get("disc_res")
        if res and res.get("fail"):
            st.rerun()
    res = st.session_state.get("disc_res")
    if not res:
        st.caption("버튼을 누르면 종목 목록과 재무를 가져와요. 한 번 가져오면 몇 시간(목록)·하루(재무) 동안 다시 쓰고, 결과는 조건을 바꿔도 바로 달라져요.")
        return
    if res.get("fail"):
        if "종목 목록" not in res["fail"]:
            st.error(res["fail"])
        return
    tbl = discover.stage2(res["tbl"], f, has_cap=not res["manual"])
    cand = tbl[tbl["통과"]].copy()
    k = st.columns(4)
    k[0].metric("전체 종목", f"{res['uni_n']:,}")
    k[1].metric("① 목록 필터 통과", f"{res['s1_n']:,}")
    k[2].metric("② 재무 조회됨", f"{res['fin_n']:,}")
    k[3].metric("② 재무 조건 통과", f"{len(cand):,}")
    st.caption(f"{res['ts']}에 불러온 데이터예요.")
    for n in res.get("notes", []):
        st.warning(n)
    if res.get("errors"):
        with st.expander(f"DART 조회 중 오류 {len(res['errors'])}건"):
            for e in res["errors"]:
                st.write(e)
    sort_label = st.selectbox("정렬", list(discover.SORTS), index=0)
    col, asc = discover.SORTS[sort_label]
    cand = cand.sort_values(col, ascending=asc, na_position="last").reset_index(drop=True)
    st.session_state.disc_names = dict(zip(tbl["종목코드"], tbl["종목명"]))
    show = cand[["종목코드", "종목명", "시가총액(조)", "ROE", "PER", "PBR", "부채비율", "영업이익률", "영업이익증가율", "영업적자횟수"]].head(100)
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "시가총액(조)": st.column_config.NumberColumn(format="%.1f"), "ROE": st.column_config.NumberColumn(format="%.1f%%"),
        "PER": st.column_config.NumberColumn(format="%.1f"), "PBR": st.column_config.NumberColumn(format="%.2f"),
        "부채비율": st.column_config.NumberColumn(format="%.0f%%"), "영업이익률": st.column_config.NumberColumn(format="%.1f%%"),
        "영업이익증가율": st.column_config.NumberColumn(format="%.0f%%"), "영업적자횟수": st.column_config.NumberColumn(format="%d")})
    if not len(cand):
        st.info("조건을 모두 통과한 종목이 없어요. 위의 발굴 조건에서 ROE·PER 같은 기준을 조금 완화해 보세요.")
        return

    if res.get("dept"):
        with st.expander("목록 진단 (소속부 분포)"):
            st.write(res["dept"])
    st.subheader("③ 상위 후보 상세 점검")
    st.caption("안전 기준(DART 공시)과 가격 흐름까지 확인해요. 종목마다 10~30초 걸리고, 가져온 데이터는 7일 동안 저장돼요.")
    n = st.slider("상위 몇 개를 점검할까요", 3, 30, min(10, max(3, min(30, len(cand)))))
    if st.button("상위 후보 상세 점검"):
        run_detail(cand["종목코드"].head(n).tolist())
    det = [c for c in st.session_state.get("disc_detail", []) if c in set(cand["종목코드"])]
    if not det:
        return
    note = admin_notice()
    if note:
        st.warning(note)
    only_safe = st.checkbox("안전 기준 미달 종목 숨기기", value=False)
    rows = []
    for c in det:
        sres = safety_for(c)
        df = hist_kr(c)
        ind = sg.indicators(df)
        if sres is None:
            continue
        items = sres[1]
        cnt = safety.summarize(items)
        if only_safe and cnt["fail"]:
            continue
        rows.append({
            "종목코드": c, "종목명": st.session_state.disc_names.get(c, c), "안전": safety.headline(items),
            "미달 항목": ", ".join(i["label"] for i in items if i["status"] == "fail") or "-",
            "확인 불가": ", ".join(i["label"] for i in items if i["status"] == "unknown") or "-",
            "목표가 여력(%)": (consensus.upside(cons_for(c)["target"], float(df["Close"].iloc[-1])) if (cons_for(c) or {}).get("has") and len(df) else np.nan),
            **dict(zip(("진입 판정", "진입 손익비", "무효가격(원)", "최대 매수(주)", "기다릴 가격(원)"), _entry_cells(c, ind))),
            "외국인 5일(억)": (st.session_state.get("flow_sum", {}).get(c) or {}).get("f5", np.nan),
            "기관 5일(억)": (st.session_state.get("flow_sum", {}).get(c) or {}).get("i5", np.nan),
            "흐름": sg.flow_of(ind)[0] if ind else "-", "52주 위치(%)": ind["pos"] if ind else np.nan,
            "RSI": ind["rsi"] if ind else np.nan, "3개월(%)": ind["m3"] if ind else np.nan})
    if not rows:
        st.info("표시할 종목이 없어요.")
        return
    rdf = pd.DataFrame(rows)
    st.dataframe(rdf, width="stretch", hide_index=True, column_config={
        "52주 위치(%)": st.column_config.NumberColumn(format="%.0f"), "RSI": st.column_config.NumberColumn(format="%.0f"),
        "3개월(%)": st.column_config.NumberColumn(format="%.1f"),
        "외국인 5일(억)": st.column_config.NumberColumn(format="%+,.0f"), "기관 5일(억)": st.column_config.NumberColumn(format="%+,.0f"),
        "목표가 여력(%)": st.column_config.NumberColumn(format="%.0f"),
        "진입 손익비": st.column_config.NumberColumn(format="%.2f"), "무효가격(원)": st.column_config.NumberColumn(format="%,d"),
        "최대 매수(주)": st.column_config.NumberColumn(format="%,d"), "기다릴 가격(원)": st.column_config.NumberColumn(format="%,d")})
    st.caption("진입 판정은 '지금 이 가격에서 새로 산다면' 구조가 괜찮은지를 봐요(손익비가 기준 이상이면 유리). 무효가격과 최대 매수 수량은 틀렸을 때 총자산의 손실 한도(규칙 탭)만 잃도록 계산한 값이에요. 후보 하나를 자세히 보려면 종목 리포트 탭에서 종목코드를 직접 입력하세요. ")
    st.caption("52주 위치가 낮을수록 1년 중 싼 구간이고, RSI가 70 이상이면 단기 과열이에요. 자세한 재무와 근거는 종목 리포트 탭에서 종목코드를 직접 입력해 확인할 수 있어요.")
    names = {f"{r['종목명']} ({r['종목코드']})": (r["종목코드"], r["종목명"]) for r in rows}
    pick = st.multiselect("관심종목에 추가", list(names))
    if st.button("관심종목에 추가") and pick:
        ok, msg = add_watch([names[p] for p in pick])
        (st.success if ok else st.error)(msg)



# ---------- 목표·성과 ----------
def current_stock_value(rules):
    """지금 보유 주식의 평가금액 합계. 시세를 하나라도 못 가져오면 (None, 못 가져온 종목)."""
    hold, kakao = st.session_state.hold, st.session_state.kakao
    pos = get_positions(hold, rules) if len(hold) else pd.DataFrame()
    kk = get_kakao(kakao) if len(kakao) else pd.DataFrame()
    missing = (pos[pos["평가금액"].isna()]["종목명"].tolist() if len(pos) else []) + (kk[kk["평가금액"].isna()]["종목명"].tolist() if len(kk) else [])
    if missing:
        return None, missing
    val = (pos["평가금액"].sum() if len(pos) else 0.0) + (kk["평가금액"].sum() if len(kk) else 0.0)
    return float(val), []



def maybe_auto_snapshot(rules):
    """앱을 연 날 하루 한 번 총자산을 자동으로 기록한다(어느 화면을 열든 시작할 때 한 번)."""
    today = dt.date.today().isoformat()
    if st.session_state.get("perf_auto") == today or st.session_state.get("perf_err") or st.session_state.get("load_error"):
        return
    if not len(st.session_state.hold):
        return
    stock_val, missing = current_stock_value(rules)
    if not stock_val or missing:
        return  # 시세를 못 가져온 종목이 있으면 자산이 작게 기록되므로 건너뛴다
    snaps_raw = st.session_state.perf_snaps
    snaps = perf.clean_snaps(snaps_raw)
    st.session_state.perf_auto = today
    if len(snaps) and (snaps["날짜"] == pd.Timestamp(today)).any():
        return
    last_cash = float(snaps["현금"].dropna().iloc[-1]) if len(snaps) and snaps["현금"].notna().any() else 0.0
    row = pd.DataFrame([[today, stock_val, last_cash, stock_val + last_cash, "자동", "현금은 마지막 기록값"]], columns=perf.SNAP_COLS)
    ok, msg = save_perf(snaps=pd.concat([snaps_view(snaps_raw), row], ignore_index=True))
    if not ok:
        st.session_state["auto_snap_err"] = msg


def tab_perf(rules):
    st.markdown("매달 총자산을 기록해서 **올해 수익률**과 **고점 대비 낙폭**을 보는 화면이에요. 총자산은 **주식 평가금액 + 현금**이고, 새로 넣은 돈(입금)은 수익으로 치지 않아요. 기록은 앱을 연 날 하루 한 번 자동으로 남아요.")
    store = get_store()
    if store is None:
        st.warning("데모 모드라서 기록이 저장되지 않아요.")
    if st.session_state.get("perf_err"):
        st.error(f"구글 시트에서 자산 기록을 읽지 못해서 저장을 막았어요. ({st.session_state['perf_err']})")
    goal = st.session_state.goal
    with st.expander("올해 목표 설정", expanded=False):
        c1, c2, c3 = st.columns(3)
        goal["min"] = c1.number_input("기대 범위 하단(%)", value=float(goal["min"]), step=0.5, key="g_min")
        goal["max"] = c2.number_input("기대 범위 상단(%)", value=float(goal["max"]), step=0.5, key="g_max")
        goal["dd"] = c3.number_input("허용 낙폭(%)", value=float(goal["dd"]), step=1.0, key="g_dd", help="고점 대비 이만큼 내려가면 경고해요")
        goal["kakao_flow"] = st.checkbox("카카오 정기매수를 외부에서 들어온 돈(입금)으로 계산", value=bool(goal["kakao_flow"]), key="g_kk",
                                         help="월급 등 계좌 밖의 돈으로 사는 거라면 켜 두세요. 이미 증권 계좌 안 현금으로 사는 거라면 끄세요.")
        st.caption("목표는 약속이 아니라 점검 기준이에요. 해마다 시장이 달라서, 범위를 벗어난 해에도 만회하려고 위험을 키우지 않는 게 중요해요.")
        if st.button("목표 저장", key="g_save"):
            ok, msg = save_settings()
            (st.success if ok else st.error)(msg)

    stock_val, missing = current_stock_value(rules)
    snaps_raw = st.session_state.perf_snaps
    snaps = perf.clean_snaps(snaps_raw)
    last_cash = float(snaps["현금"].dropna().iloc[-1]) if len(snaps) and snaps["현금"].notna().any() else 0.0
    today = pd.Timestamp(dt.date.today())

    snaps_raw = st.session_state.perf_snaps
    snaps = perf.clean_snaps(snaps_raw)
    if missing:
        st.warning("시세를 가져오지 못한 종목이 있어서 오늘 자산을 자동 기록하지 않았어요: " + ", ".join(missing))
    if st.session_state.get("auto_snap_err"):
        st.error(f"오늘 자산을 자동으로 기록하지 못했어요: {st.session_state['auto_snap_err']}")

    flows = perf.all_flows(st.session_state.perf_flows, st.session_state.kakao, goal["kakao_flow"],
                           snaps["날짜"].min() if len(snaps) else today, snaps["날짜"].max() if len(snaps) else today)
    ser = perf.build_series(snaps, flows) if len(snaps) else None
    ys = perf.year_stats(ser, today.year) if ser is not None else None

    if ys is None:
        st.info("수익률을 계산하려면 **기준이 되는 기록 하나**와 **오늘 기록**이 필요해요. 오늘 기록은 자동으로 남으니, 아래 '자산 기록 관리'에서 **연초(작년 12월 말) 총자산을 직접 한 줄 추가**해 주세요. 증권사 앱의 그날 평가금액을 합친 값이면 돼요.")
    else:
        dd = perf.drawdown(ser)
        ydd = perf.drawdown(ser, since=ys["base_date"])
        k = st.columns(4)
        k[0].metric("총자산(최근 기록)", man(ys["end_total"]), f"기록일 {ys['end_date'].date()}", delta_color="off")
        k[1].metric("올해 수익률(입금 보정)", pct(ys["twr"] * 100))
        k[2].metric("올해 손익", man(ys["profit"]))
        k[3].metric("고점 대비 낙폭", pct(dd["current"] * 100), f"올해 최대 {ydd['max'] * 100:.1f}%", delta_color="off")
        st.caption(f"{ys['base_date'].date()}부터 {ys['end_date'].date()}까지 계산했어요. 단순 증감은 {pct(ys['simple'] * 100)}이고, 그 사이 입금(+)·출금(-)은 합계 {man(ys['flow'])}이에요. 차이가 크면 입금 기록을 확인해 보세요.")
        if ys["approx_base"]:
            st.warning(f"작년 말 기록이 없어서 올해 첫 기록({ys['base_date'].date()})을 기준으로 계산했어요. 연초 이후 변화가 일부 빠졌을 수 있어요.")
        age = (today - ys["end_date"]).days
        if age >= 7:
            st.warning(f"마지막 기록이 {age}일 전이에요. 그 사이 현금이나 입출금이 바뀌었다면 아래에서 기록을 갱신해 주세요.")
        cur, lo, hi = ys["twr"] * 100, float(goal["min"]), float(goal["max"])
        state = "범위 아래" if cur < lo else ("범위 안" if cur <= hi else "범위 위")
        st.write(f"**올해 목표 범위 {lo:g}~{hi:g}%** 중 현재 **{cur:.1f}%** — {state}")
        st.progress(float(min(max(cur / hi, 0.0), 1.0)) if hi > 0 else 0.0)
        left = (pd.Timestamp(year=today.year, month=12, day=31) - today).days
        st.caption(f"올해 남은 기간은 {left}일이에요. 목표는 점검 기준일 뿐이고, 못 미쳐도 위험을 키워 만회하려 하지 않는 게 좋아요.")
        if dd["current"] * 100 <= -float(goal["dd"]):
            st.error(f"고점 대비 {dd['current'] * 100:.1f}%로 허용 낙폭(-{goal['dd']:g}%)을 넘었어요. 보유 종목을 점검하고, 산 이유가 아직 유효한지 먼저 확인해 보세요.")
        a, b = st.columns(2)
        with a:
            st.markdown("**총자산 추이**")
            st.altair_chart(alt.Chart(ser).mark_line(point=True, color="#0F6B63").encode(
                x=alt.X("날짜:T", title=None), y=alt.Y("총자산:Q", scale=alt.Scale(zero=False), title=None),
                tooltip=["날짜:T", alt.Tooltip("총자산:Q", format=",.0f")]).properties(height=240), width="stretch")
        with b:
            st.markdown("**고점 대비 낙폭**")
            sdf = dd["series"].copy()
            sdf["낙폭(%)"] = sdf["낙폭"] * 100
            st.altair_chart(alt.Chart(sdf).mark_area(color="#2A63D4", opacity=0.5).encode(
                x=alt.X("날짜:T", title=None), y=alt.Y("낙폭(%):Q", title=None),
                tooltip=["날짜:T", alt.Tooltip("낙폭(%):Q", format=".1f")]).properties(height=240), width="stretch")
        st.markdown("**월말 자산**")
        me = perf.month_end(ser).copy()
        me["월말 총자산(만원)"] = (me["월말 총자산"] / 1e4).round(0)
        me["월간 수익률(%)"] = (me["월간 수익률"] * 100).round(1)
        me["입출금(만원)"] = (me["입출금"] / 1e4).round(0)
        st.dataframe(me[["월", "월말 총자산(만원)", "월간 수익률(%)", "입출금(만원)"]].iloc[::-1], width="stretch", hide_index=True,
                     column_config={"월말 총자산(만원)": st.column_config.NumberColumn(format="%,d"), "입출금(만원)": st.column_config.NumberColumn(format="%,d")})
        st.caption("월말 자산은 그 달 마지막 기록이에요. 앱을 열지 않은 날은 기록이 없으니, 월말에는 한 번 열어서 현금도 같이 갱신해 주세요.")

    with st.expander("자산 기록 관리 (현금·연초 기준·입출금)", expanded=ys is None):
        c1, c2 = st.columns([2, 1])
        cash = c1.number_input("지금 현금(예수금 합계, 원)", min_value=0, value=int(last_cash), step=100000, key="g_cash",
                               help="증권사 계좌들의 예수금을 합친 값이에요. 총자산 = 주식 평가금액 + 현금이에요.")
        if c2.button("지금 기록하기", type="primary", key="g_rec"):
            if stock_val is None:
                st.error("시세를 가져오지 못한 종목이 있어서 기록하지 못했어요.")
            else:
                base = snaps_view(st.session_state.perf_snaps)
                base = base[base["날짜"] != str(today.date())]
                row = pd.DataFrame([[str(today.date()), stock_val, float(cash), stock_val + float(cash), "수동", ""]], columns=perf.SNAP_COLS)
                ok, msg = save_perf(snaps=pd.concat([base, row], ignore_index=True))
                (st.success if ok else st.error)(msg)
                if ok:
                    st.session_state.pop("g_snaps_ed", None)
                    st.rerun()
        st.markdown("**자산 기록** — 연초 기준을 넣을 때는 날짜(예: 2025-12-31)와 총자산만 적으면 돼요. 잘못된 줄은 삭제할 수 있어요.")
        ed = st.data_editor(snaps_view(st.session_state.perf_snaps), num_rows="dynamic", width="stretch", hide_index=True, key="g_snaps_ed",
                            column_config={"날짜": st.column_config.TextColumn(help="2026-01-31 형식"),
                                           "주식": st.column_config.NumberColumn(format="%,d"), "현금": st.column_config.NumberColumn(format="%,d"),
                                           "총자산": st.column_config.NumberColumn(format="%,d")})
        if st.button("자산 기록 저장", key="g_snaps_save"):
            ok, msg = save_perf(snaps=ed)
            (st.success if ok else st.error)(msg)
            if ok:
                st.session_state.pop("g_snaps_ed", None)
                st.rerun()
        st.markdown("**입출금** — 증권 계좌 밖에서 들어온 돈은 +, 빠져나간 돈은 -로 적어요(예: 2026-03-15, 50000000). 주식 사고팔기는 입출금이 아니에요.")
        fe = st.data_editor(flows_view(st.session_state.perf_flows), num_rows="dynamic", width="stretch", hide_index=True, key="g_flows_ed",
                            column_config={"날짜": st.column_config.TextColumn(help="2026-03-15 형식"), "금액": st.column_config.NumberColumn(format="%,d")})
        if st.button("입출금 저장", key="g_flows_save"):
            ok, msg = save_perf(flows=fe)
            (st.success if ok else st.error)(msg)
            if ok:
                st.session_state.pop("g_flows_ed", None)
                st.rerun()


def tab_safety():
    st.markdown("보유 종목의 재무·공시 데이터를 DART에서 가져와 **내 안전 기준**과 비교해요. 한 번 가져오면 구글 시트에 저장돼서 "
                f"{FRESH_DAYS}일 동안은 다시 가져오지 않아요. 안전 기준 숫자는 '규칙' 탭에서 바꿔요.")
    hold = st.session_state.hold
    if not len(hold):
        st.info("보유 종목을 먼저 등록해 주세요.")
        return
    dart, err = get_dart()
    if dart is None:
        st.warning(err)
        return
    note = admin_notice()
    if note:
        st.warning(note)
    uniq = hold.drop_duplicates("종목코드")[["종목코드", "종목명"]].to_dict("records")
    todo = [r for r in uniq if not (st.session_state.fincache.get(r["종목코드"]) and is_fresh(st.session_state.fincache[r["종목코드"]]))]
    c1, c2 = st.columns([1, 3])
    force = c2.checkbox("전부 새로 가져오기", value=False)
    if c1.button("재무·공시 가져오기", type="primary"):
        targets = uniq if force else todo
        if not targets:
            st.success("모든 종목이 최신이에요.")
        else:
            bar, errors = st.progress(0.0), []
            for i, r in enumerate(targets):
                bar.progress(i / len(targets), text=f"{r['종목명'] or r['종목코드']} 가져오는 중… ({i + 1}/{len(targets)})")
                _, e = fetch_company(r["종목코드"])
                if e:
                    errors.append(f"{r['종목명'] or r['종목코드']}: {e}")
                    if "010" in e or "011" in e or "020" in e:
                        break  # 키 문제나 한도 초과면 더 시도해도 소용없어요
            bar.progress(1.0)
            save_fincache()
            if errors:
                st.error("가져오지 못한 종목이 있어요:\n\n" + "\n\n".join(errors))
            else:
                st.rerun()
    rows = []
    for r in uniq:
        code = r["종목코드"]
        sres = safety_for(code)
        d = st.session_state.fincache.get(code)
        if sres is None:
            rows.append({"종목": r["종목명"] or code, "갱신일": "-", "결과": "미조회", "미달 항목": "-", "확인 불가": "-"})
        else:
            items = sres[1]
            bad = ", ".join(i["label"] for i in items if i["status"] == "fail") or "-"
            unk = ", ".join(i["label"] for i in items if i["status"] == "unknown") or "-"
            rows.append({"종목": r["종목명"] or code, "갱신일": d.get("fetched", "-"), "결과": safety.headline(items), "미달 항목": bad, "확인 불가": unk})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("자세한 수치와 근거는 '종목 리포트' 탭에서 종목을 고르면 볼 수 있어요. 관리종목은 거래소 목록으로 확인하고, 투자경고·투자주의 지정은 점검에 포함되지 않아요.")



def tab_journal(rules):
    st.markdown("판단할 때의 숫자(상승 여력, 하락 위험, 손익비, 필요 승률)와 이유를 **그 순간 그대로 저장**하고, 20·60·120거래일 뒤의 결과를 자동으로 붙여요. "
                "몇 달 쌓이면 **내 판단의 실제 승률과 손익비**가 보여요. 사고팔 때만이 아니라 **보유를 유지하기로 한 결정**도 남겨야 '팔지 않은 판단'의 결과를 알 수 있어요.")
    if get_store() is None:
        st.warning("데모 모드라서 기록이 저장되지 않아요.")
    if st.session_state.get("jr_err"):
        st.error(f"구글 시트에서 판단 기록을 읽지 못해서 저장을 막았어요. ({st.session_state['jr_err']})")
    hold = st.session_state.hold
    last_d, due = journal_due()
    if len(hold):
        st.write(f"마지막 기록은 **{'없음' if last_d is None else f'{last_d}일 전'}**이고, {JOURNAL_DUE_DAYS}일 넘게 기록이 없는 보유 종목은 **{len(due)}개**예요.")

    with st.expander("월간 점검: 보유 종목을 한 번에 기록", expanded=bool(due)):
        codes = list(hold["종목코드"].drop_duplicates())
        show_all = st.checkbox("점검 대상이 아닌 종목도 보기", key="jm_all")
        targets = [c for c in codes if show_all or c in due]
        if not targets:
            st.success("점검할 종목이 없어요.")
        pick = {}
        for code in targets:
            ag = holding_agg(code)
            card = card_for(code, ag["tag"], ag["avg"], ag["qty"], ag["since"]) if ag else None
            if not card:
                st.caption(f"{ag['name'] if ag else code}: 시세를 가져오지 못해서 건너뛰었어요.")
                continue
            c1, c2, c3, c4 = st.columns([3, 2, 1, 3])
            c1.markdown(f"**{ag['name']}** · {ag['tag']}  \n등급 {card['grade']} · 필요 승률 {'-' if card['need_win'] is None else str(round(card['need_win'])) + '%'} · 수익률 {pct(card['ret']) if card['ret'] is not None else '-'}")
            opts = ["기록 안 함"] + journal.DECISIONS
            dec = c2.selectbox("결정", opts, index=opts.index("보유 유지") if code in due else 0, key=f"jm_dec_{code}", label_visibility="collapsed")
            conf = c3.selectbox("확신", [1, 2, 3, 4, 5], index=2, key=f"jm_conf_{code}", label_visibility="collapsed")
            memo = c4.text_input("메모", key=f"jm_memo_{code}", label_visibility="collapsed", placeholder="한 줄 메모(선택)")
            if dec != "기록 안 함":
                pick[code] = (card, ag, dec, conf, memo)
        if targets and st.button(f"선택한 {len(pick)}건 기록하기", type="primary", key="jm_save", disabled=not pick):
            rows = []
            for code, (c, ag, dec, conf, memo) in pick.items():
                scx = score_for(code, ag["tag"], ag["avg"], ag["qty"], ag["since"], card=c)
                rows.append(make_record(c, code, ag["name"], dec, "월간 점검", conf, c.get("invalid"), memo, scx["total"] if scx else None))
            ok, msg = save_journal(rows)
            (st.success if ok else st.error)(msg)
            if ok:
                for k in [k for k in st.session_state if k.startswith(("jm_dec_", "jm_conf_", "jm_memo_"))]:
                    del st.session_state[k]
                st.rerun()
        st.caption("월간 점검의 이유는 '월간 점검'으로 저장되고, 무효가격은 카드가 제안한 값(가까운 지지선 아래)이 들어가요. 개별 종목을 자세히 기록하려면 종목 리포트 탭의 카드 아래를 쓰세요.")

    jr = st.session_state.jr
    if jr is None or jr.empty:
        st.info("아직 기록이 없어요. 위에서 첫 기록을 남겨 보세요. 결과는 20거래일(약 한 달) 뒤부터 나와요.")
        return
    codes = jr["종목코드"].drop_duplicates().tolist()
    closes = {}
    for c in codes:
        d = hist_kr(c)
        if len(d):
            closes[c] = d["Close"]
    bidx = hist_index()
    ev = journal.evaluate(jr, closes, bidx["Close"] if len(bidx) else None)
    st.subheader("내 판단의 결과")
    st.session_state.setdefault("jr_h", 60)
    h = st.radio("몇 거래일 뒤 결과로 볼까요", list(journal.HORIZONS), horizontal=True, format_func=lambda x: f"{x}일(약 {x // 20}개월)", key="jr_h")
    done = int(ev[f"v{h}"].notna().sum())
    st.caption(f"기록 {len(ev)}건 중 {h}일이 지나 결과가 나온 것은 {done}건이에요. 방향은 이렇게 봐요: 사거나 들고 있는 결정은 이후 오르면 맞은 판단, 팔거나 사지 않은 결정은 이후 내리면 맞은 판단이에요(손익은 그 방향으로 계산).")
    if done:
        for title, by in (("결정 유형별", None), ("확신도별", "확신구분"), ("기록 당시 카드 등급별", "등급"), ("기록 당시 종합 점수대별", "점수구간"), ("이유별", "이유")):
            e2 = ev.copy()
            e2["확신구분"] = e2["확신도"].map(journal.confidence_bucket)
            e2["점수구간"] = e2["종합점수"].map(score.bucket)
            e2.loc[e2["점수구간"] == "-", "점수구간"] = np.nan
            s = journal.summarize(e2, h, by)
            if len(s) > 1 or by is None:
                st.markdown(f"**{title}**")
                st.dataframe(s, width="stretch", hide_index=True, column_config={
                    "맞은 비율(%)": st.column_config.NumberColumn(format="%.0f"), "평균 이익(%)": st.column_config.NumberColumn(format="%+.1f"),
                    "평균 손실(%)": st.column_config.NumberColumn(format="%+.1f"), "손익비": st.column_config.NumberColumn(format="%.2f"),
                    "건당 기대값(%)": st.column_config.NumberColumn(format="%+.1f")})
        st.caption(f"건수가 {journal.MIN_N}건보다 적으면 우연일 가능성이 커서 '참고만'으로 표시해요. 맞은 비율보다 **건당 기대값**(평균 이익과 평균 손실을 합친 값)이 더 중요해요. 확신도가 높은 쪽이 더 잘 맞는지, 카드가 '유리'라고 한 쪽이 실제로 나았는지도 여기서 확인해요.")
    else:
        st.info(f"아직 {h}일이 지난 기록이 없어요. 아래 표의 '현재까지(%)'로 진행 상황은 볼 수 있어요.")
    with st.expander("기록 전체 보기", expanded=True):
        t = ev.sort_values("날짜", ascending=False).copy()
        t["날짜"] = t["날짜"].dt.strftime("%Y-%m-%d")
        cols = ["날짜", "종목명", "결정", "이유", "확신도", "등급", "필요승률", "기록가", "현재까지(%)", "경과일", "v20", "v60", "v120", "a60", "mae60", "inv60", "메모"]
        t = t[[c for c in cols if c in t.columns]].rename(columns={"v20": "20일 결정손익(%)", "v60": "60일 결정손익(%)", "v120": "120일 결정손익(%)",
                                                                  "a60": "60일 코스피 대비(%)", "mae60": "60일 최대 하락(%)", "inv60": "60일 내 무효가격 이탈", "필요승률": "필요 승률(%)"})
        st.dataframe(t, width="stretch", hide_index=True, column_config={
            "기록가": st.column_config.NumberColumn(format="%,d"), "현재까지(%)": st.column_config.NumberColumn(format="%+.1f"),
            "20일 결정손익(%)": st.column_config.NumberColumn(format="%+.1f"), "60일 결정손익(%)": st.column_config.NumberColumn(format="%+.1f"),
            "120일 결정손익(%)": st.column_config.NumberColumn(format="%+.1f"), "60일 코스피 대비(%)": st.column_config.NumberColumn(format="%+.1f"),
            "60일 최대 하락(%)": st.column_config.NumberColumn(format="%.1f"), "필요 승률(%)": st.column_config.NumberColumn(format="%.0f")})



# ---------- 목적 자금 ----------
def stock_value_parts():
    """주식 잔고(평가금액) 합계: (보유 종목, 카카오 정기매수). 시세를 못 가져온 종목은 뺀다."""
    hold, kakao, rules = st.session_state.hold, st.session_state.kakao, st.session_state.rules
    a_ = float(get_positions(hold, rules)["평가금액"].sum(skipna=True)) if len(hold) else 0.0
    b_ = float(get_kakao(kakao)["평가금액"].sum(skipna=True)) if len(kakao) else 0.0
    return a_, b_


def tab_fund():
    st.markdown("노후·자녀 분가처럼 **금액과 시점이 정해진 돈**이 목표에 닿으려면 **연 몇 %의 수익이 필요한지** 계산해요. 이 앱은 **주식 투자금액만** 관리하므로, "
                "**현재 금액은 이 앱의 주식 잔고(평가금액)를 합산해서 자동으로** 쓰고, 목적마다 **배분 비율**로 나눠요. 이 숫자는 예측이 아니라 **필요한 조건**이에요.")
    goal = st.session_state.goal
    a_, b_ = stock_value_parts()
    c1, c2 = st.columns(2)
    inc_cash = c2.checkbox("증권 계좌 현금(목표·성과에서 마지막으로 기록한 값)도 투자금액에 포함", value=False, key="fund_cash",
                           help="기본은 주식 평가금액만 써요. 투자하려고 계좌에 둔 현금까지 목적 자금으로 보고 싶을 때 켜세요.")
    snaps = perf.clean_snaps(st.session_state.get("perf_snaps", pd.DataFrame(columns=perf.SNAP_COLS)))
    cash = float(snaps["현금"].dropna().iloc[-1]) if (inc_cash and len(snaps) and snaps["현금"].notna().any()) else 0.0
    total = a_ + b_ + cash
    k0 = st.columns(4)
    k0[0].metric("현재 주식 투자금액", f"{total / 1e4:,.0f}만원", "잔고 합산(자동)", delta_color="off",
                 help="내 자산 화면의 보유 종목 평가금액과 카카오 정기매수 평가금액을 합친 값이에요. 시세에 따라 매일 바뀌고, 아래 목적별 현재 금액에 자동으로 반영돼요.")
    k0[1].metric("보유 종목", f"{a_ / 1e4:,.0f}만원")
    k0[2].metric("카카오 정기매수", f"{b_ / 1e4:,.0f}만원")
    k0[3].metric("계좌 현금(포함 시)", f"{cash / 1e4:,.0f}만원" if inc_cash else "미포함")
    if total <= 0:
        st.warning("주식 평가금액을 계산하지 못했어요. 보유 종목을 입력하고 시세가 들어오면 자동으로 채워져요.")
    goal["inflation"] = c1.number_input("물가 상승률(연 %)", min_value=0.0, value=float(goal.get("inflation", 2.5)), step=0.5, key="g_infl",
                                        help="목표 금액을 오늘 가치로 적었을 때, 그 가치를 지키려면 미래에 더 큰 금액이 필요해요.")
    goal["infl_on"] = st.checkbox("목표 금액을 오늘 가치로 보고 물가만큼 키워서 계산", value=bool(goal.get("infl_on", True)), key="g_inflon")

    fdf = pd.DataFrame(st.session_state.funds)
    if "매달 저축(만원)" in fdf.columns and "매달 투입(만원)" not in fdf.columns:      # 옛 형식으로 저장된 값을 새 열 이름으로 옮긴다
        fdf = fdf.rename(columns={"매달 저축(만원)": "매달 투입(만원)"})
    if "배분(%)" not in fdf.columns:
        fdf["배분(%)"] = 100.0 / max(len(fdf), 1)
    for c in FUND_COLS:
        if c not in fdf.columns:
            fdf[c] = "" if c == "이름" else 0.0
    st.markdown("**목적별 목표** — 금액은 **만원 단위**예요. 목표는 오늘 가치로 적고, **배분(%)**은 위의 주식 투자금액 중 그 목적에 속한 비율이에요(합계 100% 이하). "
                "**매달 투입**은 매달 주식에 새로 넣는 돈이에요.")
    ed = st.data_editor(fdf[FUND_COLS], num_rows="dynamic", width="stretch", hide_index=True, key="fund_editor", column_config={
        "목표(만원)": st.column_config.NumberColumn(min_value=0, step=100, format="%,d"), "기간(년)": st.column_config.NumberColumn(min_value=0.5, step=0.5),
        "배분(%)": st.column_config.NumberColumn(min_value=0, max_value=100, step=5, format="%.0f", help="현재 주식 투자금액 중 이 목적에 배정하는 비율이에요."),
        "매달 투입(만원)": st.column_config.NumberColumn(min_value=0, step=10, format="%,d")})
    if st.button("목적 자금 저장", type="primary", key="fund_save"):
        st.session_state.funds = ed.fillna(0).to_dict("records")
        ok, msg = save_settings()
        (st.success if ok else st.error)(msg)
    rows_ = ed.fillna(0).to_dict("records")
    share_sum = sum(float(r["배분(%)"]) for r in rows_)
    scale = 100.0 / share_sum if share_sum > 100 else 1.0
    if share_sum > 100:
        st.warning(f"배분 합계가 {share_sum:.0f}%로 100%를 넘었어요. 계산에서는 100%에 맞춰 비율을 줄여서 썼어요.")
    elif share_sum < 100 and share_sum > 0:
        st.caption(f"배분 합계가 {share_sum:.0f}%예요. 나머지 {100 - share_sum:.0f}%({total * (100 - share_sum) / 100 / 1e4:,.0f}만원)는 어느 목적에도 속하지 않은 투자금액이에요.")
    infl = goal["inflation"] / 100 if goal.get("infl_on", True) else 0.0
    shown = 0
    for row in rows_:
        tgt, yrs, mo = float(row["목표(만원)"]), float(row["기간(년)"]), float(row["매달 투입(만원)"])
        share = float(row["배분(%)"]) * scale
        cur = total / 1e4 * share / 100
        if tgt <= 0 or yrs <= 0:
            continue
        shown += 1
        adj = fund.future_target(tgt, infl, yrs)
        r = fund.required_return(adj, cur, mo, yrs)
        lvl, desc = fund.level(r)
        st.subheader(f"{row['이름'] or '이름 없음'}")
        k = st.columns(5)
        k[0].metric("목표(오늘 가치)", f"{tgt:,.0f}만원", f"{yrs:g}년 뒤", delta_color="off")
        k[1].metric("그때 필요한 금액", f"{adj:,.0f}만원", None if infl == 0 else f"물가 {goal['inflation']:.1f}% 반영", delta_color="off")
        k[2].metric("현재 배분 금액", f"{cur:,.0f}만원", f"배분 {share:.0f}% · 목표의 {cur / adj * 100:.0f}%" if adj else None, delta_color="off",
                    help="주식 잔고에 배분 비율을 곱한 값이에요. 시세에 따라 매일 바뀌어요.")
        k[3].metric("필요한 연 수익률", "불가능" if r is None else ("0% 이하" if r <= 0 else f"{r * 100:.1f}%"),
                    help="지금 배정된 금액과 매달 투입을 이 수익률로 굴리면 목표에 닿는다는 뜻이에요. 매년 이 수익이 나온다는 보장은 없어요.")
        k[4].metric("부담 정도", lvl)
        st.write(desc)
        rows = []
        for rt in fund.SCENARIOS:
            end = fund.fv(cur, mo, rt, yrs)
            need_mo = fund.required_monthly(adj, cur, rt, yrs)
            rows.append({"가정 수익률": f"연 {rt * 100:.0f}%", f"{yrs:g}년 뒤 예상 금액(만원)": end, "목표 대비(만원)": end - adj,
                         "목표에 닿는 매달 투입(만원)": need_mo})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            f"{yrs:g}년 뒤 예상 금액(만원)": st.column_config.NumberColumn(format="%,d"), "목표 대비(만원)": st.column_config.NumberColumn(format="%+,d"),
            "목표에 닿는 매달 투입(만원)": st.column_config.NumberColumn(format="%,d")})
        lever = []
        for add in (0, 10, 30, 50):
            rr = fund.required_return(adj, cur, mo + add, yrs)
            lever.append({"매달 투입": f"{mo + add:,.0f}만원" + ("" if add == 0 else f" (+{add})"), "필요한 연 수익률": "불가능" if rr is None else ("0% 이하" if rr <= 0 else f"{rr * 100:.1f}%")})
        st.markdown("**매달 투입을 늘리면 필요한 수익률이 이렇게 내려가요**")
        st.dataframe(pd.DataFrame(lever), width="stretch", hide_index=True)
    if not shown:
        st.info("위 표에 목표 금액과 기간을 적고 저장하면 계산해 드려요.")
    st.caption("수익률 숫자는 가정이에요. 실제 수익은 해마다 크게 달라지고 손실이 날 수도 있어요. 이 앱은 주식 투자금액만 보기 때문에 예금·부동산·연금 등 다른 자산은 반영하지 않아요. "
               "주식에 얼마를 둘지(자산 배분)는 종목 선택보다 큰 결정이라서, 인증된 재무설계사와 상담해서 정하는 것을 권해요. 이 앱은 투자 자문이 아니에요.")


# ---------- 매매 계획 ----------
def tab_plan(rules):
    st.markdown("사기 전이나 보유 중에 **무효가격, 1·2차 목표와 정리 비율, 보호폭**을 미리 정해두고, 앱이 **현재 가격과 비교해서 점검 시점을 알려줘요.** "
                "타이밍을 맞히는 게 아니라 **미리 정한 기준에 닿았을 때 감정이 아니라 계획으로 판단**하게 하는 도구예요. 정리는 앱이 대신 하지 않고, 계획대로 실행하면 직접 '완료'로 표시하세요.")
    if st.session_state.get("plans_err"):
        st.error(f"구글 시트에서 매매 계획을 읽지 못해서 저장을 막았어요. ({st.session_state['plans_err']})")
    if get_store() is None:
        st.warning("데모 모드라서 계획이 저장되지 않아요.")
    rows = plan_rows(rules)
    if rows:
        tbl = []
        for p, ev in rows:
            st_ = {s["단계"].split("(")[0]: s for s in ev["steps"]}
            tbl.append({"종목": p["종목명"] or p["종목코드"], "현재가": ev["price"], "기준가 대비(%)": ev["pl_pct"], "점검 결과": ev["action"],
                        "무효가격까지(%)": st_.get("무효가격", {}).get("현재가 대비(%)"), "1차 목표까지(%)": st_.get("1차 목표", {}).get("현재가 대비(%)"),
                        "2차 목표까지(%)": st_.get("2차 목표", {}).get("현재가 대비(%)")})
        st.dataframe(pd.DataFrame(tbl), width="stretch", hide_index=True, column_config={
            "현재가": st.column_config.NumberColumn(format="%,d"), "기준가 대비(%)": st.column_config.NumberColumn(format="%+.1f"),
            "무효가격까지(%)": st.column_config.NumberColumn(format="%+.1f"), "1차 목표까지(%)": st.column_config.NumberColumn(format="%+.1f"),
            "2차 목표까지(%)": st.column_config.NumberColumn(format="%+.1f")})
    else:
        st.info("진행 중인 계획이 없어요. 종목 리포트의 진입 평가에서 '이 조건으로 매매 계획 만들기'를 누르거나, 아래에서 새로 만드세요.")

    for p, ev in rows:
        icon = {"warn": "⚠️", "info": "🔔", "ok": "✅"}[ev["level"]]
        with st.expander(f"{icon} {p['종목명'] or p['종목코드']} — {ev['action']}", expanded=ev["level"] == "warn"):
            if ev["price"] is not None:
                st.dataframe(pd.DataFrame(ev["steps"]), width="stretch", hide_index=True, column_config={
                    "가격": st.column_config.NumberColumn(format="%,d"), "현재가 대비(%)": st.column_config.NumberColumn(format="%+.1f")})
                st.caption(f"계획을 만든 날({p['생성일'].date()}) 이후 최고가 {ev['peak']:,.0f}원, 현재가 {ev['price']:,.0f}원")
            pid = p["id"]
            c1, c2, c3, c4 = st.columns(4)
            inv = c1.number_input("무효가격", min_value=0.0, value=float(p["무효가격"] or 0), step=100.0, key=f"pl_inv_{pid}")
            t1 = c2.number_input("1차 목표", min_value=0.0, value=float(p["목표1"] or 0), step=100.0, key=f"pl_t1_{pid}")
            t2 = c3.number_input("2차 목표", min_value=0.0, value=float(p["목표2"] or 0), step=100.0, key=f"pl_t2_{pid}")
            w = c4.number_input("보호폭(%)", min_value=0.0, value=float(p["보호폭"] or 0), step=1.0, key=f"pl_w_{pid}")
            d1, d2, d3 = st.columns([1, 1, 3])
            r1 = d1.number_input("1차 정리 비율(%)", min_value=0.0, max_value=100.0, value=float(p["비율1"] or 0), step=5.0, key=f"pl_r1_{pid}")
            r2 = d2.number_input("2차 정리 비율(%)", min_value=0.0, max_value=100.0, value=float(p["비율2"] or 0), step=5.0, key=f"pl_r2_{pid}")
            memo = d3.text_input("메모", value=p["메모"], key=f"pl_memo_{pid}")
            b1, b2, b3, b4 = st.columns(4)
            pl = st.session_state.plans.copy()
            idx = pl.index[pl["id"] == pid]
            if b1.button("수정 저장", key=f"pl_save_{pid}"):
                pl.loc[idx, ["무효가격", "목표1", "목표2", "보호폭", "비율1", "비율2", "메모"]] = [inv, t1, t2, w, r1, r2, memo]
                ok, msg = save_plans(pl)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
            if b2.button("1차 정리 완료 표시", key=f"pl_d1_{pid}", disabled=p["완료1"] == "Y"):
                pl.loc[idx, "완료1"] = "Y"
                ok, msg = save_plans(pl)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
            if b3.button("2차 정리 완료 표시", key=f"pl_d2_{pid}", disabled=p["완료2"] == "Y"):
                pl.loc[idx, "완료2"] = "Y"
                ok, msg = save_plans(pl)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
            if b4.button("계획 종료", key=f"pl_end_{pid}"):
                pl.loc[idx, "상태"] = "종료"
                ok, msg = save_plans(pl)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
            st.caption("정리를 실행했다면 판단 기록 탭에도 한 줄 남겨 두면, 계획대로 했을 때의 결과를 나중에 비교할 수 있어요.")

    with st.expander("새 계획 만들기"):
        hold = st.session_state.hold
        opts = {f"{r['종목명'] or r['종목코드']} ({r['종목코드']}) · 보유": r["종목코드"] for r in hold.drop_duplicates("종목코드").to_dict("records")}
        for w_ in st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS)).to_dict("records"):
            opts.setdefault(f"{w_['종목명'] or w_['종목코드']} ({w_['종목코드']}) · 관심", w_["종목코드"])
        opts["직접 입력"] = ""
        pick = st.selectbox("종목", list(opts), key="np_pick")
        code = opts[pick] or st.text_input("종목코드(6자리)", key="np_code").strip().zfill(6)
        ag = holding_agg(code) if code else None
        df_ = hist_kr(code) if code else pd.DataFrame()
        if code and len(df_):
            tag = (ag or {}).get("tag", "중장기")
            card = card_for(code, tag, (ag or {}).get("avg"), (ag or {}).get("qty"), (ag or {}).get("since"))
            price = float(df_["Close"].iloc[-1])
            st.caption(f"현재가 {price:,.0f}원" + ("" if not card else " · 아래 기본값은 종합 판단 카드가 제안한 값이에요(바꿔도 돼요)."))
            e = entry_for(card, price) if card else None
            c1, c2, c3 = st.columns(3)
            base = c1.number_input("기준가(내 평단 또는 살 가격)", min_value=0.0, value=float((ag or {}).get("avg") or price), step=100.0, key=f"np_base_{code}")
            qty = c2.number_input("수량", min_value=0.0, value=float((ag or {}).get("qty") or (e["shares"] if e and e["shares"] else 0)), step=1.0, key=f"np_qty_{code}")
            tg = c3.selectbox("꼬리표", ["중장기", "스윙"], index=0 if tag == "중장기" else 1, key=f"np_tag_{code}")
            d1, d2, d3, d4 = st.columns(4)
            inv = d1.number_input("무효가격", min_value=0.0, value=float(round(e["invalid"])) if e and e["invalid"] else 0.0, step=100.0, key=f"np_inv_{code}")
            t1 = d2.number_input("1차 목표", min_value=0.0, value=float(round(card["up_price"])) if card and card.get("up_price") else float(round(price * 1.15)), step=100.0, key=f"np_t1_{code}")
            cons = cons_for(code)
            t2d = (cons["target"] if cons and cons.get("has") and cons.get("target") else (card["ups"][0]["price"] if card and card["ups"] else price * 1.3))
            t2 = d3.number_input("2차 목표", min_value=0.0, value=float(round(t2d)), step=100.0, key=f"np_t2_{code}")
            w = d4.number_input("보호폭(%)", min_value=0.0, value=float(rules.get(tg, {}).get("보호폭", 15)), step=1.0, key=f"np_w_{code}")
            f1, f2, f3 = st.columns([1, 1, 3])
            r1 = f1.number_input("1차 정리 비율(%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0, key=f"np_r1_{code}")
            r2 = f2.number_input("2차 정리 비율(%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0, key=f"np_r2_{code}")
            memo = f3.text_input("메모(계획의 이유)", key=f"np_memo_{code}")
            if st.button("계획 저장", type="primary", key=f"np_save_{code}"):
                pl = st.session_state.plans
                if not inv or inv >= price and not ag:
                    st.error("무효가격은 현재가보다 낮게 정해 주세요.")
                elif len(pl[(pl["종목코드"] == code) & (pl["상태"] == "진행")]):
                    st.warning("이 종목은 이미 진행 중인 계획이 있어요. 위에서 수정하거나 종료한 뒤 새로 만드세요.")
                else:
                    name = (ag or {}).get("name") or (card and code) or code
                    row = {"id": dt.datetime.now().strftime("%Y%m%d%H%M%S") + code, "종목코드": code, "종목명": name, "생성일": dt.date.today().isoformat(), "상태": "진행",
                           "꼬리표": tg, "기준가": base, "수량": qty, "무효가격": inv, "목표1": t1, "비율1": r1, "목표2": t2, "비율2": r2, "보호폭": w, "완료1": "", "완료2": "", "메모": memo}
                    ok, msg = save_plans(pd.concat([pl, pd.DataFrame([row])], ignore_index=True))
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()
        elif code:
            st.warning("시세를 가져오지 못했어요. 종목코드를 확인하세요.")



# ---------- 가설 실험실 ----------
@st.cache_resource
def _lab_store():
    return {}


def load_hyp():
    if "hyp" in st.session_state:
        return
    df, err = pd.DataFrame(columns=HYP_COLS), None
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            df = store.read("가설결과", HYP_COLS)
        except Exception as e:
            err = str(e)
    st.session_state.hyp, st.session_state.hyp_err = df, err


def save_hyp(rows):
    store = get_store()
    if store is not None and (st.session_state.get("load_error") or st.session_state.get("hyp_err")):
        return False, "구글 시트에서 가설 결과를 읽지 못한 상태라 저장을 막았어요."
    df = pd.concat([st.session_state.hyp, pd.DataFrame(rows, columns=HYP_COLS)], ignore_index=True).tail(600)
    try:
        if store is not None:
            store.write("가설결과", df)
        st.session_state.hyp = df
        return True, "결과를 구글 시트에 저장했어요."
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"


def evidence_for(signal_title):
    """종목 리포트의 신호가 우리 검증에서 어떻게 나왔는지(가장 최근 결과). 없으면 빈 문자열."""
    name = lab.SIGNAL_TO_HYP.get(signal_title)
    h = st.session_state.get("hyp")
    if not name or h is None or h.empty:
        return ""
    g = h[h["가설"] == name]
    if g.empty:
        return ""
    r = g.iloc[-1]
    return f" → **우리 검증({r['실행일']}, {r['종목수']}종목·{r['기간(년)']}년·{r['보유(일)']}일):** {r['등급']} (시장 대비 평균 {float(r['평균초과수익(%)']):+.2f}%)"


def lab_universe(n):
    """검증에 쓸 종목코드. 시가총액 상위 n개(우선주·스팩 제외). 종목 목록을 못 받으면 보유·관심 종목."""
    try:
        uni = listing_cached()
        uni = uni[[discover.is_common_stock(c, nm) for c, nm in zip(uni["Code"], uni["Name"])]]
        return list(uni.sort_values("Marcap", ascending=False)["Code"].head(n)), None
    except Exception as e:
        codes = list(st.session_state.hold["종목코드"]) + list(st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS))["종목코드"])
        return list(dict.fromkeys(codes)), f"종목 목록을 못 받아서 보유·관심 종목({len(set(codes))}개)으로만 검증해요: {dart_data.redact(e)[:80]}"


def lab_prices(codes, years, bar):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    store, days = _lab_store(), int(years * 366) + 60
    need = [c for c in codes if (c, years) not in store or time.time() - store[(c, years)][0] > 86400]
    done = 0
    if need:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(market.history_kr, c, days): c for c in need}
            for f in as_completed(futs):
                try:
                    store[(futs[f], years)] = (time.time(), f.result())
                except Exception:
                    store[(futs[f], years)] = (time.time(), pd.DataFrame(columns=["Close"]))
                done += 1
                bar.progress(done / len(need), text=f"과거 가격을 가져오는 중… ({done}/{len(need)})")
    return {c: store[(c, years)][1] for c in codes if (c, years) in store}


def lab_panel(codes, years, h, bar):
    key = (tuple(codes), years, h)
    cached = _lab_store().get("panel")
    if cached and cached[0] == key:
        return cached[1]
    prices = lab_prices(codes, years, bar)
    bar.progress(1.0, text="지표와 신호를 계산하는 중…")
    P = lab.build(prices, h)
    _lab_store()["panel"] = (key, P)
    return P


def load_flowlog():
    if "flowlog" in st.session_state:
        return
    df, err = pd.DataFrame(columns=FLOWLOG_COLS), None
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            df = store.read("수급기록", FLOWLOG_COLS)
        except Exception as e:
            err = str(e)
    for c in FLOWLOG_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    st.session_state.flowlog, st.session_state.flowlog_err = df, err


def save_flowlog(codes, depth=40, bar=None):
    """종목들의 최근 수급(depth 거래일)을 '수급기록' 탭에 이어 붙인다. 이미 있는 날짜는 건너뛴다.
    반환: (성공, 문구, 오류 목록, 가장 긴 기간 행 수)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    store = get_store()
    if store is not None and (st.session_state.get("load_error") or st.session_state.get("flowlog_err")):
        return False, "구글 시트에서 수급 기록을 읽지 못한 상태라 저장을 막았어요.", [], 0
    cur = st.session_state.flowlog
    have = set(zip(cur["날짜"].astype(str), cur["종목코드"]))
    pages = max(2, int(np.ceil(depth / 20)))
    results, errs = {}, []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(flows.fetch_flow, c, pages): c for c in codes}
        for i, f in enumerate(as_completed(futs)):
            c = futs[f]
            try:
                results[c] = f.result()
            except Exception as e:
                errs.append(f"{c}: {str(e)[:100]}")
            if bar:
                bar.progress((i + 1) / max(len(codes), 1), text=f"수급을 가져오는 중… ({i + 1}/{len(codes)})")
    new, longest = [], 0
    for c, d in results.items():
        longest = max(longest, len(d))
        for r in d.itertuples():
            day = r.날짜.strftime("%Y-%m-%d")
            if (day, c) in have:
                continue
            new.append([day, c, r.기관, r.외국인, getattr(r, "개인", np.nan), r.종가, getattr(r, "외국인보유율", np.nan)])
    if not new:
        return True, "새로 추가할 날짜가 없어요(이미 모두 기록돼 있어요).", errs, longest
    df = pd.concat([cur, pd.DataFrame(new, columns=FLOWLOG_COLS)], ignore_index=True).sort_values(["종목코드", "날짜"]).tail(150000)
    try:
        if store is not None:
            store.write("수급기록", df)
        st.session_state.flowlog = df
        return True, f"{len(new):,}줄을 추가했어요.", errs, longest
    except Exception as e:
        return False, f"저장에 실패했어요: {e}", errs, longest


def tab_lab():
    st.markdown("**모든 지표는 가설이에요.** 20일선, 볼린저 밴드, 이격도, RSI 같은 흔한 지표가 한국 종목에서 실제로 효과가 있었는지 과거 가격으로 재봐요. "
                "여기서 통과한 것만 판단 점수에 쓰고, 통과하지 못한 것은 '참고(가설)'로만 봐요. **51%를 찾아 1%p씩 올리려면 먼저 무엇이 효과가 있는지 가려내야 해요.**")
    with st.expander("검증 원칙 (꼭 읽어 보세요)", expanded=False):
        st.markdown(
            "- **표본이 중요해요.** 승률 50%와 51%를 구분하려면 약 1만 건이 필요해요. 그래서 한 종목이 아니라 수백 종목 × 수년의 이벤트를 모아요.\n"
            "- **시장 평균을 뺀 초과수익**으로 비교해요. 시장 전체가 오른 날의 착시를 없애기 위해서예요.\n"
            "- **같은 종목의 겹치는 날은 한 번만** 세요(보유 기간 간격을 둬요).\n"
            "- **여러 가설을 동시에 시험하면 우연히 맞는 게 나와요.** 시험한 개수만큼 기준을 엄격하게 올려요(다중검정 보정).\n"
            "- **앞 절반과 뒤 절반 기간의 방향이 같아야** 믿어요.\n"
            "- **결과에 한계가 있어요.** 지금 상장된 종목만 쓰기 때문에 상장폐지된 종목이 빠져 있어서(생존 편향) 결과가 실제보다 좋게 나올 수 있어요. 과거에 효과가 있었다는 것이 앞으로도 있다는 보장은 아니에요.")
    c1, c2, c3, c4 = st.columns(4)
    n_uni = c1.slider("종목 수(시가총액 상위)", 30, 300, 120, 10, key="lab_n")
    years = c2.radio("기간", [3, 5], horizontal=True, key="lab_y", format_func=lambda x: f"최근 {x}년")
    h = c3.radio("보유 기간", [20, 60], horizontal=True, key="lab_h", format_func=lambda x: f"{x}거래일")
    cost = c4.number_input("왕복 비용(%)", min_value=0.0, value=0.3, step=0.05, key="lab_cost", help="세금과 수수료를 합친 값이에요. 이 비용을 뺀 수익으로 판단해요.")
    names = st.multiselect("검증할 가설", list(lab.CATALOG), default=list(lab.CATALOG), key="lab_names")
    if st.button("가설 검증 실행", type="primary", key="lab_run"):
        if not names:
            st.error("가설을 하나 이상 고르세요.")
        else:
            bar = st.progress(0.0, text="준비 중…")
            codes, note = lab_universe(n_uni)
            if note:
                st.warning(note)
            P = lab_panel(codes, years, h, bar)
            if P is None:
                bar.empty()
                st.error("검증할 가격 자료를 가져오지 못했어요. 잠시 뒤 다시 시도해 주세요.")
            else:
                res = {}
                for i, k in enumerate(names):
                    bar.progress(i / len(names), text=f"{k} 검증 중…")
                    res[k] = lab.event_study(P["S"][k], P, cost / 100, n_tests=len(names))
                bar.empty()
                st.session_state["lab_res"] = {"res": res, "meta": (len(P["C"].columns), years, h, cost, len(names)), "date": dt.date.today().isoformat()}
                rows = [[dt.date.today().isoformat(), k, len(P["C"].columns), years, h, o.get("n", 0), o.get("win_diff"), o.get("mean_excess"), o.get("ci_lo"), o.get("ci_hi"),
                         "Y" if o.get("consistent") else "N", o["grade"]] for k, o in res.items()]
                ok, msg = save_hyp(rows)
                if not ok:
                    st.warning(msg)
    lr = st.session_state.get("lab_res")
    if lr:
        nu, yy, hh, cc, kk = lr["meta"]
        res = lr["res"]
        st.subheader(f"결과 — {nu}종목 · 최근 {yy}년 · {hh}거래일 보유 · 비용 {cc}%")
        tbl = []
        for k, o in res.items():
            tbl.append({"가설": k, "설명": lab.CATALOG[k][0], "표본": o.get("n"), "승률(%)": o.get("win"), "기준선 승률(%)": o.get("win_base"), "승률 차이(%p)": o.get("win_diff"),
                        "평균 초과수익(%)": o.get("mean_excess"), "신뢰구간(하한)": o.get("ci_lo"), "신뢰구간(상한)": o.get("ci_hi"), "앞 절반": o.get("half1"),
                        "뒤 절반": o.get("half2"), "평균 순수익(%)": o.get("mean_net"), "평균 이익(%)": o.get("avg_win"), "평균 손실(%)": o.get("avg_loss"), "손익비": o.get("payoff"),
                        "평균 최대 하락(%)": o.get("mean_mae"), "등급": o["grade"]})
        df = pd.DataFrame(tbl).sort_values("평균 초과수익(%)", ascending=False, na_position="last")
        st.dataframe(df, width="stretch", hide_index=True, column_config={
            "승률(%)": st.column_config.NumberColumn(format="%.1f"), "기준선 승률(%)": st.column_config.NumberColumn(format="%.1f"), "승률 차이(%p)": st.column_config.NumberColumn(format="%+.1f"),
            "평균 초과수익(%)": st.column_config.NumberColumn(format="%+.2f"), "신뢰구간(하한)": st.column_config.NumberColumn(format="%+.2f"), "신뢰구간(상한)": st.column_config.NumberColumn(format="%+.2f"),
            "앞 절반": st.column_config.NumberColumn(format="%+.2f"), "뒤 절반": st.column_config.NumberColumn(format="%+.2f"), "평균 최대 하락(%)": st.column_config.NumberColumn(format="%.1f"),
            "평균 순수익(%)": st.column_config.NumberColumn(format="%+.2f", help="비용을 뺀 건당 평균 수익(기대값)이에요."), "평균 이익(%)": st.column_config.NumberColumn(format="%+.1f"),
            "평균 손실(%)": st.column_config.NumberColumn(format="%+.1f"), "손익비": st.column_config.NumberColumn(format="%.2f", help="평균 이익 ÷ 평균 손실(절댓값)이에요.")})
        bo = next((o for o in res.values() if "base_mean_net" in o), None)
        if bo:
            bw_ = next((o.get("win_base") for o in res.values() if o.get("win_base") is not None), None)
            bpay = (bo["base_avg_win"] / abs(bo["base_avg_loss"])) if (bo["base_avg_win"] == bo["base_avg_win"] and bo["base_avg_loss"] == bo["base_avg_loss"] and bo["base_avg_loss"]) else float("nan")
            st.success(f"**기준선(아무 종목이나 {hh}거래일 보유)**: 승률 {bw_:.1f}% · 건당 평균 순수익 {bo['base_mean_net']:+.2f}% · 평균 이익 {bo['base_avg_win']:+.1f}% · 평균 손실 {bo['base_avg_loss']:+.1f}% · 손익비 {bpay:.2f}. "
                       "신호의 성적은 이 기준선과 비교하세요. 승률이 50%보다 낮아도 평균 이익이 평균 손실보다 충분히 크면 기대값은 플러스일 수 있어요.")
        mdes = [o["mde"] for o in res.values() if "mde" in o]
        if mdes:
            st.info(f"이 표본으로는 승률이 **약 {np.median(mdes):.1f}%p 이상 달라야** 우연과 구분돼요. 그보다 작은 차이는 '효과가 없다'가 아니라 '**구분이 안 된다**'는 뜻이에요. "
                    f"{kk}개를 동시에 시험해서 신뢰구간은 **{(1 - 0.05 / max(1, kk)) * 100:.1f}% 수준**으로 엄격하게 잡았어요.")
        st.caption("승률은 비용을 뺀 수익이 플러스인 비율이에요. '평균 초과수익'은 같은 날 같은 기간의 시장(검증 종목 평균) 대비 얼마나 더 올랐는지예요. 등급: 검증됨(약함)은 신뢰구간이 0보다 크고 앞뒤 기간이 일관된 경우, 가설(방향만)은 방향은 맞지만 우연과 구분되지 않는 경우, 효과 구분 안 됨은 차이를 가려내지 못한 경우예요.")

        st.markdown("**교차 검증 — 가설을 조합해 보기** (두세 개를 동시에 만족할 때)")
        pick = st.multiselect("조합할 가설(2~3개)", list(lab.CATALOG), max_selections=3, key="lab_combo")
        if len(pick) >= 2 and st.button("조합 검증", key="lab_combo_run"):
            codes, _ = lab_universe(nu)
            cached = _lab_store().get("panel")
            P = cached[1] if cached and cached[0][1:] == (yy, hh) else None
            if P is None:
                st.error("위 검증을 다시 실행한 뒤 조합해 주세요.")
            else:
                st.session_state["lab_combo_n"] = st.session_state.get("lab_combo_n", 0) + 1
                o = lab.event_study(lab.combine(P, pick), P, cc / 100, n_tests=kk + st.session_state["lab_combo_n"])
                st.session_state["lab_combo_res"] = (" + ".join(pick), o)
        cr = st.session_state.get("lab_combo_res")
        if cr:
            o = cr[1]
            if "mean_excess" in o:
                st.write(f"**{cr[0]}**: 표본 {o['n']:,}건 · 승률 {o['win']:.1f}% (기준선 {o['win_base']:.1f}%) · 평균 초과수익 {o['mean_excess']:+.2f}% "
                         f"(신뢰구간 {o['ci_lo']:+.2f}~{o['ci_hi']:+.2f}) · 앞 {o['half1']:+.2f} / 뒤 {o['half2']:+.2f} → **{o['grade']}**")
            else:
                st.write(f"**{cr[0]}**: 표본이 부족해요({o['n']}건).")
            st.caption("조합을 여러 번 시험할수록 우연히 좋은 결과가 나오기 쉬워서, 시험한 횟수만큼 기준을 더 엄격하게 올려요.")

    st.divider()
    st.subheader("요인 검증 — 종목의 성격이 이후 수익과 관련 있었나")
    st.markdown("매번 전 종목을 **요인 값 순서로 5등분**해서, 윗그룹이 아랫그룹보다 이후 수익이 더 컸는지 비교해요(모멘텀, 단기 반전, 변동성, 52주 위치, 거래대금). "
                "신호가 켜진 날만 보는 위 검증과 달리 **모든 종목을 한꺼번에 비교**해서 더 힘 있게 가려내요. 위의 종목 수·기간·보유 기간 설정을 그대로 써요.")
    if st.button("요인 검증 실행", key="lab_factor_run"):
        bar = st.progress(0.0, text="준비 중…")
        codes, note = lab_universe(n_uni)
        if note:
            st.warning(note)
        P = lab_panel(codes, years, h, bar)
        if P is None or "V" not in P:
            bar.empty()
            st.error("검증할 가격 자료를 가져오지 못했어요. 잠시 뒤 다시 시도해 주세요.")
        else:
            fres = {}
            for i, (k, (_, fn)) in enumerate(lab.FACTORS.items()):
                bar.progress(i / len(lab.FACTORS), text=f"{k} 검증 중…")
                fres[k] = lab.factor_study(fn(P["C"], P["V"]), P, n_tests=len(lab.FACTORS))
            bar.empty()
            st.session_state["lab_fres"] = {"res": fres, "meta": (len(P["C"].columns), years, h)}
            ok, msg = save_hyp([[dt.date.today().isoformat(), "요인: " + k, len(P["C"].columns), years, h, o.get("n_dates", 0), None, o.get("spread"), o.get("ci_lo"), o.get("ci_hi"),
                                 "Y" if o.get("consistent") else "N", o["grade"]] for k, o in fres.items()])
            if not ok:
                st.warning(msg)
    fr = st.session_state.get("lab_fres")
    if fr:
        nu, yy, hh = fr["meta"]
        st.markdown(f"**{nu}종목 · 최근 {yy}년 · {hh}거래일 보유 · 5일마다 5등분**")
        rows = []
        for k, o in fr["res"].items():
            r = {"요인": k, "설명": lab.FACTORS[k][0], "날짜 수": o.get("n_dates"), "윗-아랫(%)": o.get("spread"), "신뢰구간(하한)": o.get("ci_lo"), "신뢰구간(상한)": o.get("ci_hi"),
                 "앞 절반": o.get("half1"), "뒤 절반": o.get("half2"), "단조성": o.get("mono"), "등급": o["grade"]}
            if "q" in o:
                for g in range(5):
                    r[f"{g + 1}분위(%)"] = o["q"][g]
            rows.append(r)
        fdf = pd.DataFrame(rows)
        st.dataframe(fdf, width="stretch", hide_index=True, column_config={
            "윗-아랫(%)": st.column_config.NumberColumn(format="%+.2f"), "신뢰구간(하한)": st.column_config.NumberColumn(format="%+.2f"), "신뢰구간(상한)": st.column_config.NumberColumn(format="%+.2f"),
            "앞 절반": st.column_config.NumberColumn(format="%+.2f"), "뒤 절반": st.column_config.NumberColumn(format="%+.2f"), "단조성": st.column_config.NumberColumn(format="%+.2f"),
            **{f"{g}분위(%)": st.column_config.NumberColumn(format="%+.2f") for g in range(1, 6)}})
        st.caption("1분위는 요인 값이 가장 낮은 그룹, 5분위는 가장 높은 그룹이에요. '윗-아랫'은 5분위 − 1분위의 평균 이후 수익(한 번 보유할 때마다)이고, 단조성이 +1에 가까우면 분위가 올라갈수록 수익도 일정하게 올라간다는 뜻이에요. "
                   "비용은 반영하지 않았어요(실제로 사고팔면 비용이 들고 윗그룹·아랫그룹을 동시에 거래하는 것은 개인이 하기 어려워요). 등급의 의미는 위 가설 검증과 같고, 생존 편향 한계도 같아요.")
        sel = st.selectbox("분위별 평균 수익 보기", [k for k, o in fr["res"].items() if "q" in o], key="lab_fsel") if any("q" in o for o in fr["res"].values()) else None
        if sel:
            q = fr["res"][sel]["q"]
            cdf = pd.DataFrame({"분위": [f"{g + 1}분위" for g in range(5)], "평균 이후 수익(%)": q})
            st.altair_chart(alt.Chart(cdf).mark_bar().encode(x=alt.X("분위:N", title=None, sort=None), y=alt.Y("평균 이후 수익(%):Q", title=None),
                                                           color=alt.condition(alt.datum["평균 이후 수익(%)"] > 0, alt.value("#D93A33"), alt.value("#2A63D4")),
                                                           tooltip=["분위", alt.Tooltip("평균 이후 수익(%):Q", format=".2f")]).properties(height=220), width="stretch")

    st.divider()
    st.subheader("수급 가설 검증 — 외국인·기관 수급이 이후 수익과 관련 있었나")
    st.markdown("한국 시장은 외국인·기관 수급의 영향이 크다는 가설을 **같은 방식(5등분 비교)**으로 재요. 다만 **수급의 과거 기록이 있어야** 하는데, 네이버가 주는 건 종목마다 최근 일부뿐이에요. "
                "그래서 아래에서 **수급을 구글 시트의 `수급기록` 탭에 쌓고**, 쌓인 만큼만 검증해요. 가격 지표가 3년치로 검증되는 것과 달리 수급은 **기록한 날부터**만 가능해요.")
    if st.session_state.get("flowlog_err"):
        st.error(f"구글 시트에서 수급 기록을 읽지 못해서 저장을 막았어요. ({st.session_state['flowlog_err']})")
    hcodes = list(dict.fromkeys(list(st.session_state.hold["종목코드"]) + list(st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS))["종목코드"])))
    d1, d2 = st.columns(2)
    depth = d1.selectbox("한 번에 가져올 과거 기간", [40, 120, 250, 500], index=0, key="fl_depth", format_func=lambda x: f"최근 {x}거래일",
                         help="길게 고를수록 한 번에 많이 쌓여요. 네이버가 그만큼 주는지는 아래 '기간 시험'으로 확인하세요.")
    extra_n = d2.slider("검증용으로 시가총액 상위 종목도 함께 기록", 0, 200, 0, 10, key="fl_extra",
                        help="5등분 비교를 하려면 종목이 수십 개 이상 필요해요. 위의 '검증할 종목 수'와 같은 기준(시가총액 상위)으로 고르세요. 많을수록 오래 걸려요.")
    t1, t2, _ = st.columns([1, 1, 2])
    if t1.button("기간 시험 (1종목)", key="flowlog_test", help="네이버가 한 종목에 대해 얼마나 긴 기간을 주는지 확인해요."):
        probe = (hcodes[0] if hcodes else "005930")
        try:
            dd = flows.fetch_flow(probe, pages=25)
            st.success(f"{probe}: 요청한 500거래일 중 **{len(dd)}거래일**({dd['날짜'].min().date()} ~ {dd['날짜'].max().date()})을 받았어요.")
        except Exception as e:
            st.error(f"시험에 실패했어요: {dart_data.redact(e)[:160]}")
    if t2.button("수급 기록하기", key="flowlog_save"):
        extra, note = (lab_universe(extra_n) if extra_n else ([], None))
        if note:
            st.warning(note)
        codes = list(dict.fromkeys(hcodes + extra))
        bar = st.progress(0.0, text="준비 중…")
        ok, msg, errs, longest = save_flowlog(codes, depth, bar)
        bar.empty()
        (st.success if ok else st.error)(msg + (f" (종목당 가장 길게 받은 기간: {longest}거래일, 대상 {len(codes)}종목)" if longest else ""))
        if longest and longest < depth * 0.8:
            st.info(f"요청한 {depth}거래일보다 짧게 받았어요. 네이버가 이 이상은 주지 않는 것으로 보여요. 앞으로 한 달에 한 번 이상 기록해서 이어 붙이세요.")
        for e in errs[:3]:
            st.warning(f"못 가져온 종목: {e}")
    fl = st.session_state.flowlog
    if len(fl):
        per = fl.groupby("종목코드")["날짜"].nunique()
        st.write(f"기록된 종목 **{per.size}개**, 종목당 최대 **{int(per.max())}거래일**, 전체 {len(fl):,}줄이에요.")
    else:
        st.info("아직 기록이 없어요. 위 버튼을 눌러 쌓기 시작하세요.")
    if st.button("수급 가설 검증 실행", key="flow_factor_run"):
        if fl.empty:
            st.error("수급 기록이 없어요. 먼저 기록하세요.")
        else:
            bar = st.progress(0.0, text="준비 중…")
            codes, note = lab_universe(n_uni)
            if note:
                st.warning(note)
            P = lab_panel(codes, years, h, bar)
            if P is None:
                bar.empty()
                st.error("검증할 가격 자료를 가져오지 못했어요.")
            else:
                n_ov, n_days, first, ok_days = lab.flow_readiness(fl, P["C"])
                F = lab.flow_matrices(fl, P["C"])
                res_f = {}
                for i, (k, M) in enumerate(F.items()):
                    bar.progress(i / len(F), text=f"{k} 검증 중…")
                    res_f[k] = lab.factor_study(M, P, n_tests=len(F))
                bar.empty()
                st.session_state["lab_flow_res"] = {"res": res_f, "ready": (n_ov, n_days, first, ok_days), "meta": (len(P["C"].columns), years, h)}
                save_hyp([[dt.date.today().isoformat(), "수급: " + k, len(P["C"].columns), years, h, o.get("n_dates", 0), None, o.get("spread"), o.get("ci_lo"), o.get("ci_hi"),
                           "Y" if o.get("consistent") else "N", o["grade"]] for k, o in res_f.items()])
    fr2 = st.session_state.get("lab_flow_res")
    if fr2:
        n_ov, n_days, first, ok_days = fr2["ready"]
        st.markdown(f"**검증 대상과 수급 기록이 겹치는 종목 {n_ov}개 · 기록 {n_days}거래일 (시작 {first or '-'}) · 종목 30개 이상이 있는 날 {ok_days}일**")
        if ok_days < 150:
            st.info(f"수급 가설을 믿을 만하게 재려면 종목 30개 이상이 **150거래일(약 7개월) 이상** 있어야 해요. 지금은 {ok_days}일이라 결과는 참고만 하세요. 기간을 길게 한 번에 가져오거나 시간이 지나면 검증력이 생겨요.")
        rows = []
        for k, o in fr2["res"].items():
            r = {"수급 요인": k, "설명": lab.FLOW_FACTORS[k], "날짜 수": o.get("n_dates"), "윗-아랫(%)": o.get("spread"), "신뢰구간(하한)": o.get("ci_lo"), "신뢰구간(상한)": o.get("ci_hi"),
                 "앞 절반": o.get("half1"), "뒤 절반": o.get("half2"), "등급": o["grade"]}
            if "q" in o:
                for g_ in range(5):
                    r[f"{g_ + 1}분위(%)"] = o["q"][g_]
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "윗-아랫(%)": st.column_config.NumberColumn(format="%+.2f"), "신뢰구간(하한)": st.column_config.NumberColumn(format="%+.2f"), "신뢰구간(상한)": st.column_config.NumberColumn(format="%+.2f"),
            "앞 절반": st.column_config.NumberColumn(format="%+.2f"), "뒤 절반": st.column_config.NumberColumn(format="%+.2f"),
            **{f"{g_}분위(%)": st.column_config.NumberColumn(format="%+.2f") for g_ in range(1, 6)}})
        st.caption("5분위는 해당 수급이 가장 많이 순매수된 종목 그룹이에요. 윗-아랫이 +이고 신뢰구간이 0보다 크면 '수급이 많이 들어온 종목이 이후 더 올랐다'는 뜻이에요. "
                   "'큰손'·'외국인'·'기관' 결과는 같은 데이터를 다르게 자른 것이라 서로 비슷하게 나올 수 있어요(독립된 확인이 아니에요). 개인 순매수는 거래 3주체의 합이 0에 가까워서 큰손 순매수와 반대로 움직이는 경향이 있어요. 생존 편향 등 한계는 위와 같아요.")


def load_scorelog():
    if "scorelog" in st.session_state:
        return
    df, err = pd.DataFrame(columns=SCORELOG_COLS), None
    store = get_store()
    if store is not None and not st.session_state.get("load_error"):
        try:
            df = store.read("점수기록", SCORELOG_COLS)
        except Exception as e:
            err = str(e)
    for c in SCORELOG_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    st.session_state.scorelog, st.session_state.scorelog_err = df, err


def record_scores(rows):
    """오늘 점수를 '점수기록' 탭에 이어 붙인다(같은 날짜·종목은 건너뜀). 반환: (성공, 문구)"""
    store = get_store()
    if store is not None and (st.session_state.get("load_error") or st.session_state.get("scorelog_err")):
        return False, "구글 시트에서 점수 기록을 읽지 못한 상태라 저장을 막았어요."
    cur = st.session_state.scorelog
    have = set(zip(cur["날짜"].astype(str), cur["종목코드"]))
    new = [r for r in rows if (r[0], r[1]) not in have]
    if not new:
        return True, "오늘 점수는 이미 기록돼 있어요."
    df = pd.concat([cur, pd.DataFrame(new, columns=SCORELOG_COLS)], ignore_index=True).tail(8000)
    try:
        if store is not None:
            store.write("점수기록", df)
        st.session_state.scorelog = df
        return True, f"{len(new)}종목의 오늘 점수를 기록했어요."
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"



def score_flags(sc):
    flags = []
    if any(i["항목"] == "개인 주도 경고" for f in sc["families"] for i in f["items"]):
        flags.append("개인 주도")
    if any(i["항목"] == "안전 기준" and i["점수"] < 0 for f in sc["families"] for i in f["items"]):
        flags.append("안전 미달")
    if sc["conflicts"]:
        flags.append(f"근거 엇갈림 {len(sc['conflicts'])}쌍")
    return flags


def prev_score(code, days=5):
    """점수 기록에서 days일 이상 전의 가장 최근 점수. 없으면 None."""
    sl = st.session_state.get("scorelog")
    if sl is None or sl.empty:
        return None
    cut = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    g = sl[(sl["종목코드"] == code) & (sl["날짜"].astype(str) <= cut)].sort_values("날짜")
    return float(g["종합점수"].iloc[-1]) if len(g) and g["종합점수"].iloc[-1] == g["종합점수"].iloc[-1] else None


VERDICT_ORDER = ["매도·비중 축소 검토", "수익실현 검토", "추가매수 검토", "지켜보기", "보유 유지"]


def tab_brief(rules):
    st.markdown("**이번 주에 무엇부터 볼지**를 한 장으로 모았어요. 점수, 신호, 매매 계획, 기록 상태를 합쳐서 우선순위로 정리하고, **10분 안에 읽도록** 만들었어요. "
                "점검할 곳을 알려줄 뿐이고, 사고팔지는 직접 판단하세요.")
    hold = st.session_state.hold
    watch = st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS))
    if not len(hold):
        st.info("보유 종목을 먼저 입력하세요(내 자산 화면).")
        return
    codes_all = list(dict.fromkeys(list(hold["종목코드"]) + list(watch["종목코드"])))
    if st.button("수급·컨센서스 새로 가져오기 (보유·관심 종목)", key="brief_fetch"):
        with st.spinner("수급과 컨센서스를 가져오는 중이에요(조금 걸려요)…"):
            e1 = fetch_flows_bulk(codes_all)
            e2 = fetch_cons_bulk(codes_all)
        st.session_state["flow_msg"] = e1 + [f"컨센서스 {e}" for e in e2]
        st.rerun()
    for e in st.session_state.pop("flow_msg", []) or []:
        st.warning(f"못 가져왔어요 — {e}")
    pos = get_positions(hold, rules)
    total_assets = sizing_assets()
    plans = {}
    for p, ev in plan_rows(rules):
        cur = plans.get(p["종목코드"])
        order = {"warn": 0, "info": 1, "ok": 2}
        if cur is None or order[ev["level"]] < order[cur[1]]:
            plans[p["종목코드"]] = (ev["action"], ev["level"])
    holds, missing_price = [], 0
    for code in hold["종목코드"].drop_duplicates():
        ag = holding_agg(code)
        rows_ = pos[pos["종목코드"] == code]
        d_ = hist_kr(code)
        if not len(d_) or rows_.empty:
            missing_price += 1
            continue
        price = float(d_["Close"].iloc[-1])
        value = float(rows_["평가금액"].sum(skipna=True))
        card = card_for(code, ag["tag"], ag["avg"], ag["qty"], ag["since"])
        sc = score_for(code, ag["tag"], ag["avg"], ag["qty"], ag["since"], card=card)
        verdicts = [v for v in rows_["판단"].tolist() if v in VERDICT_ORDER]
        verdict = min(verdicts, key=VERDICT_ORDER.index) if verdicts else "보유 유지"
        sigs = []
        for s_ in rows_["신호"].tolist():
            sigs += [x.strip() for x in str(s_).split(",") if x.strip() and x.strip() != "-"]
        plan_a = plans.get(code)
        holds.append({"name": ag["name"], "code": code, "weight": (value / total_assets * 100) if total_assets else None,
                      "ret": ((price / ag["avg"] - 1) * 100) if ag["avg"] else None, "score": sc["total"] if sc else None,
                      "score_prev": prev_score(code), "n_avail": sc["n_avail"] if sc else 0, "verdict": verdict, "signals": list(dict.fromkeys(sigs)),
                      "flags": score_flags(sc) if sc else [], "plan_level": plan_a[1] if plan_a else None, "plan_action": plan_a[0] if plan_a else None})
    cands = []
    for code in watch["종목코드"]:
        if code in set(hold["종목코드"]):
            continue
        d_ = hist_kr(code)
        if not len(d_):
            continue
        price = float(d_["Close"].iloc[-1])
        card = card_for(code, "중장기", None, None, None)
        sc = score_for(code, "중장기", None, None, None, card=card)
        if not card or not sc:
            continue
        e = entry_for(card, price)
        nm = watch[watch["종목코드"] == code]["종목명"].iloc[0] or code
        cands.append({"name": nm, "code": code, "score": sc["total"], "n_avail": sc["n_avail"], "entry_verdict": e["verdict"].split("(")[0], "ratio": e["ratio"],
                      "invalid": e["invalid"], "shares": e["shares"], "wait_price": e["wait_price"]})
    last_d, due = journal_due()
    fl = st.session_state.get("flowlog")
    fl_days = None if fl is None or fl.empty else int((pd.Timestamp.today().normalize() - pd.to_datetime(fl["날짜"]).max()).days)
    snaps = perf.clean_snaps(st.session_state.get("perf_snaps", pd.DataFrame(columns=perf.SNAP_COLS)))
    manual = snaps[snaps["구분"] != "자동"] if len(snaps) else snaps
    cash_days = int((pd.Timestamp.today().normalize() - manual["날짜"].max()).days) if len(manual) else None
    fc = st.session_state.get("fincache", {})
    stale = sum(1 for c in hold["종목코드"].drop_duplicates() if c not in fc or (dt.date.today() - dt.date.fromisoformat(str(fc[c].get("fetched", "2000-01-01")))).days > FRESH_DAYS)
    top = max(holds, key=lambda h: h["weight"] or 0) if holds else None
    res = brief.build(holds, cands, {"due_n": len(due), "flowlog_days": fl_days, "cash_days": cash_days, "stale_fin_n": stale, "missing_price_n": missing_price},
                      {"top_name": top["name"] if top else None, "top_weight": top["weight"] if top else None, "max_pos": st.session_state.risk.get("max_pos")})
    st.subheader(f"이번 주 브리핑 — {dt.date.today().isoformat()}")
    k = st.columns(4)
    k[0].metric("먼저 확인", f"{res['counts'][1]}건")
    k[1].metric("이번 주 안에", f"{res['counts'][2]}건")
    k[2].metric("여유 있을 때", f"{res['counts'][3]}건")
    k[3].metric("매수 후보", f"{len(res['cands'])}개", help="관심종목 중 종합 점수 55점 이상, 근거 3가지 이상, 진입 구조가 유리·보통인 종목이에요(최대 3개).")
    top_rows = [a for a in res["actions"] if a["우선순위"] <= 2]
    if top_rows:
        st.dataframe(pd.DataFrame(top_rows)[["구분", "종목", "내용", "볼 곳"]], width="stretch", hide_index=True)
    else:
        st.success("이번 주에 급하게 볼 것은 없어요.")
    low_rows = [a for a in res["actions"] if a["우선순위"] == 3]
    if low_rows:
        with st.expander(f"여유 있을 때 확인 ({len(low_rows)}건)"):
            st.dataframe(pd.DataFrame(low_rows)[["종목", "내용", "볼 곳"]], width="stretch", hide_index=True)

    st.markdown("**보유 종목 현황** (비중은 총자산 기준)")
    if holds:
        order = {"매도·비중 축소 검토": 0, "수익실현 검토": 1, "추가매수 검토": 2, "지켜보기": 3, "보유 유지": 4}
        hdf = pd.DataFrame([{"종목": h["name"], "비중(%)": h["weight"], "수익률(%)": h["ret"], "종합점수": h["score"], "지난주 대비": (h["score"] - h["score_prev"]) if h["score"] is not None and h["score_prev"] is not None else None,
                             "근거": f"{h['n_avail']}/5", "판정": h["verdict"], "주의": ", ".join(h["flags"]), "계획": h["plan_action"] or "-", "_o": order.get(h["verdict"], 5)} for h in holds])
        hdf = hdf.sort_values(["_o", "비중(%)"], ascending=[True, False]).drop(columns=["_o"])
        st.dataframe(hdf, width="stretch", hide_index=True, column_config={
            "비중(%)": st.column_config.NumberColumn(format="%.1f"), "수익률(%)": st.column_config.NumberColumn(format="%+.1f"),
            "종합점수": st.column_config.NumberColumn(format="%.0f"), "지난주 대비": st.column_config.NumberColumn(format="%+.0f", help="5일 이상 전에 기록한 점수와의 차이예요. 점수 기록이 쌓이면 나와요.")})
    if res["cands"]:
        st.markdown("**매수 후보** — 종합 점수가 높고 틀렸을 때의 손실 한도 안에서 살 수 있는 관심종목")
        st.dataframe(pd.DataFrame([{"종목": c["name"], "종합점수": c["score"], "진입 손익비": c["ratio"], "무효가격(원)": c["invalid"], "최대 매수(주)": c["shares"]} for c in res["cands"]]),
                     width="stretch", hide_index=True, column_config={"종합점수": st.column_config.NumberColumn(format="%.0f"), "진입 손익비": st.column_config.NumberColumn(format="%.2f"),
                                                                       "무효가격(원)": st.column_config.NumberColumn(format="%,d"), "최대 매수(주)": st.column_config.NumberColumn(format="%,d")})
        st.caption("후보는 '사라'는 뜻이 아니라 먼저 볼 만한 종목이에요. 종목 리포트의 진입 평가에서 구조를 확인하고, 사기로 정했다면 매매 계획을 먼저 만드세요.")
    elif not len(watch):
        st.caption("관심종목(종목 발굴에서 추가)이 있어야 매수 후보를 골라줘요.")
    with st.expander("텍스트로 복사해서 보관하기"):
        st.code(brief.to_text(res, dt.date.today().isoformat()), language=None)


def tab_rank(rules):
    st.markdown("**흩어진 근거를 하나의 점수로 모아서** 한눈에 비교해요. 가격 흐름, 수급, 재무·안전, 컨센서스, 진입 구조(손익비)를 각각 -100~+100으로 매기고 가중치(규칙 탭)로 모은 종합 점수예요(50 중립). "
                "점수는 **이길 확률이 아니라 근거가 얼마나 한 방향으로 모이는지**예요. 근거 칸이 3/5 미만이면 참고만 하세요.")
    hold, watch = st.session_state.hold, st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS))
    kinds = {}
    for c in hold["종목코드"].drop_duplicates():
        kinds[c] = "보유"
    for c in watch["종목코드"]:
        kinds.setdefault(c, "관심")
    extra = st.text_input("함께 비교할 종목코드(쉼표로 구분, 선택)", key="rank_extra", placeholder="예: 005930, 000660")
    for c in [x.strip().zfill(6) for x in extra.split(",") if x.strip()]:
        kinds.setdefault(c, "추가")
    codes = list(kinds)
    if not codes:
        st.info("보유 종목이나 관심종목을 먼저 입력하세요.")
        return
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("수급 가져오기", key="rank_flow"):
        with st.spinner("수급을 가져오는 중이에요…"):
            errs = fetch_flows_bulk(codes)
        st.session_state["flow_msg"] = errs
        st.rerun()
    if b2.button("컨센서스 가져오기", key="rank_cons"):
        with st.spinner("컨센서스를 가져오는 중이에요…"):
            errs = fetch_cons_bulk(codes)
        st.session_state["flow_msg"] = [f"컨센서스 {e}" for e in errs]
        st.rerun()
    for e in st.session_state.pop("flow_msg", []) or []:
        st.warning(f"못 가져왔어요 — {e}")
    rows, log = [], []
    today = dt.date.today().isoformat()
    for code in codes:
        ag = holding_agg(code)
        tag = ag["tag"] if ag else "중장기"
        card = card_for(code, tag, ag["avg"] if ag else None, ag["qty"] if ag else 0.0, ag["since"] if ag else None)
        sc = score_for(code, tag, ag["avg"] if ag else None, ag["qty"] if ag else 0.0, ag["since"] if ag else None, card=card)
        df_ = hist_kr(code)
        if sc is None or not len(df_):
            continue
        price = float(df_["Close"].iloc[-1])
        fam = {f["name"]: f["score"] for f in sc["families"]}
        flags = score_flags(sc)
        name = (ag or {}).get("name") or (watch[watch["종목코드"] == code]["종목명"].iloc[0] if (watch["종목코드"] == code).any() else code)
        rows.append({"종목": name, "코드": code, "구분": kinds[code], "종합점수": sc["total"], "판정": sc["band"], "가격 흐름": fam["가격 흐름"], "수급": fam["수급"],
                     "재무·안전": fam["재무·안전"], "컨센서스": fam["컨센서스"], "진입 구조": fam["진입 구조"], "근거": f"{sc['n_avail']}/{sc['n_total']}",
                     "일치": "-" if not sc["agree"] or sc["agree"][1] == 0 else f"{sc['agree'][0]}/{sc['agree'][1]}", "주의": ", ".join(flags),
                     "수익률(%)": (price / ag["avg"] - 1) * 100 if ag and ag["avg"] else np.nan, "_n": sc["n_avail"]})
        if sc["total"] is not None:
            log.append([today, code, sc["total"], fam["가격 흐름"], fam["수급"], fam["재무·안전"], fam["컨센서스"], fam["진입 구조"], sc["n_avail"], price])
    if not rows:
        st.warning("시세를 가져오지 못해서 점수를 계산하지 못했어요.")
        return
    view = st.radio("보기", ["전체 순위", "보유 종목 점검(점수 낮은 순)", "매수 후보(미보유, 점수 높은 순)"], horizontal=True, key="rank_view")
    df = pd.DataFrame(rows)
    if view.startswith("보유"):
        df = df[df["구분"] == "보유"].sort_values("종합점수", ascending=True)
    elif view.startswith("매수"):
        df = df[df["구분"] != "보유"].sort_values("종합점수", ascending=False)
    else:
        df = df.sort_values("종합점수", ascending=False)
    cc = {c: st.column_config.NumberColumn(format="%+.0f") for c in ("가격 흐름", "수급", "재무·안전", "컨센서스", "진입 구조")}
    cc["종합점수"] = st.column_config.NumberColumn(format="%.0f")
    cc["수익률(%)"] = st.column_config.NumberColumn(format="%+.1f")
    st.dataframe(df.drop(columns=["_n"]), width="stretch", hide_index=True, column_config=cc)
    low = int((df["_n"] < 3).sum())
    if low:
        st.caption(f"근거가 3가지 미만인 종목이 {low}개 있어요. 위 버튼으로 수급·컨센서스를 가져오고, 재무는 안전 점검 탭에서 가져오면 근거가 늘어요.")
    st.caption("각 칸은 근거 종류별 점수(-100 불리 ~ +100 우호)이고 비어 있으면 그 근거가 없는 거예요. 점수가 높다고 사라는 뜻도, 낮다고 팔라는 뜻도 아니에요. 점수는 아래 기록으로 실제 성과와 비교해서 고쳐 나가요.")

    ok_log = [r for r in log if r[8] >= 3]
    sl0 = st.session_state.scorelog
    logged_today = set(sl0[sl0["날짜"].astype(str) == today]["종목코드"]) if sl0 is not None and len(sl0) else set()
    todo = [r for r in ok_log if r[1] not in logged_today]
    if todo:
        ok, msg = record_scores(todo)
        st.caption(f"오늘의 점수를 자동으로 기록했어요(근거 3가지 이상인 종목만). {msg}" if ok else msg)
    else:
        st.caption("오늘 점수는 기록돼 있어요(근거 3가지 이상인 종목만 기록해요). 수급·컨센서스를 늦게 가져온 종목은 다음에 이 화면을 열 때 기록돼요.")

    with st.expander("점수의 실제 성과 확인 (기록이 쌓이면)"):
        sl = st.session_state.scorelog
        if sl is None or sl.empty:
            st.info("아직 점수 기록이 없어요. 이 화면을 여는 날마다 근거 3가지 이상인 종목의 점수가 자동으로 쌓여요.")
        else:
            res = []
            for r in sl.to_dict("records"):
                d = hist_kr(r["종목코드"])
                if not len(d) or r["종합점수"] != r["종합점수"]:
                    continue
                s = d["Close"]
                i0 = int(s.index.searchsorted(pd.to_datetime(r["날짜"]), side="right")) - 1
                if i0 < 0 or i0 + 20 >= len(s) or not r["종가"] or r["종가"] != r["종가"]:
                    continue
                res.append({"구간": score.bucket(r["종합점수"]), "수익률": (float(s.iloc[i0 + 20]) / float(r["종가"]) - 1) * 100})
            n_all = len(sl)
            st.write(f"기록 {n_all:,}건 중 20거래일이 지나 결과가 나온 것은 **{len(res):,}건**이에요.")
            if res:
                rd = pd.DataFrame(res)
                g = rd.groupby("구간")["수익률"].agg(건수="count", 평균수익률="mean", 승률=lambda x: (x > 0).mean() * 100).reindex(["65 이상", "55~65", "45~55", "35~45", "35 미만"]).dropna(how="all").reset_index()
                st.dataframe(g, width="stretch", hide_index=True, column_config={"평균수익률": st.column_config.NumberColumn(format="%+.2f"), "승률": st.column_config.NumberColumn(format="%.0f")})
                st.caption("점수가 높은 구간의 이후 20거래일 수익이 실제로 더 컸는지 보는 표예요. 같은 날 같은 종목이 겹치고 시장 전체의 움직임도 섞여 있어서, 건수가 수백 건이 되기 전에는 우연일 수 있어요. 가중치를 바꾸기 전에 이 표와 판단 기록의 점수대별 통계를 같이 보세요.")


def tab_rules():
    st.markdown("규칙 숫자를 바꾸면 신호가 바로 달라져요. 바꾼 뒤 맨 아래 '규칙 저장'을 눌러야 구글 시트에 저장돼요.")
    r = st.session_state.rules
    a, b = st.columns(2)
    with a:
        st.markdown("**중장기**")
        r["중장기"]["점검선"] = st.number_input("점검선(손실 %)", value=r["중장기"]["점검선"], step=1.0)
        r["중장기"]["목표"] = st.number_input("목표수익률(%)", value=r["중장기"]["목표"], step=1.0, key="l_t")
        r["중장기"]["고점권"] = st.number_input("52주 고점권 기준(%)", value=r["중장기"]["고점권"], step=1.0)
        r["중장기"]["보호활성"] = st.number_input("수익 보호 시작(고점 기준 수익률 % 이상)", value=float(r["중장기"]["보호활성"]), step=1.0, key="l_ta",
                                            help="고점에서 이만큼 이상 수익이 난 적이 있어야 보호선이 켜져요.")
        r["중장기"]["보호폭"] = st.number_input("수익 보호폭(고점 대비 % 하락)", value=float(r["중장기"]["보호폭"]), step=1.0, key="l_tw",
                                           help="고점 대비 이만큼 내려오면 수익 보호선 도달이에요.")
        r["중장기"]["지지선"] = int(st.checkbox("지지선 이탈 신호 사용", value=bool(r["중장기"].get("지지선", 1)), key="l_sp",
                                          help="과거에 반등했던 가격대를 종가로 이탈하면 점검 신호를 줘요."))
        r["중장기"]["보호알림일"] = int(st.number_input("이탈 알림 유지 기간(거래일)", min_value=1, value=int(r["중장기"].get("보호알림일", 10)), step=1, key="l_td",
                                                help="보호선을 이탈한 뒤 이 기간 동안만 판단에 반영해요. 지나면 이탈 뒤 고점 기준으로 보호선이 다시 잡혀요."))
    with b:
        st.markdown("**스윙**")
        r["스윙"]["손절선"] = st.number_input("손절선(%)", value=r["스윙"]["손절선"], step=0.5)
        r["스윙"]["목표"] = st.number_input("목표수익률(%) ", value=r["스윙"]["목표"], step=1.0, key="s_t")
        r["스윙"]["RSI과열"] = st.number_input("RSI 과열(이상)", value=r["스윙"]["RSI과열"], step=1.0)
        r["스윙"]["눌림_하단"] = st.number_input("눌림 RSI 하단", value=r["스윙"]["눌림_하단"], step=1.0)
        r["스윙"]["눌림_상단"] = st.number_input("눌림 RSI 상단", value=r["스윙"]["눌림_상단"], step=1.0)
        r["스윙"]["보호활성"] = st.number_input("수익 보호 시작(고점 기준 수익률 % 이상) ", value=float(r["스윙"]["보호활성"]), step=1.0, key="s_ta")
        r["스윙"]["보호폭"] = st.number_input("수익 보호폭(고점 대비 % 하락) ", value=float(r["스윙"]["보호폭"]), step=1.0, key="s_tw")
        r["스윙"]["지지선"] = int(st.checkbox("지지선 이탈 신호 사용 ", value=bool(r["스윙"].get("지지선", 1)), key="s_sp"))
        r["스윙"]["보호알림일"] = int(st.number_input("이탈 알림 유지 기간(거래일) ", min_value=1, value=int(r["스윙"].get("보호알림일", 10)), step=1, key="s_td"))
    st.divider()
    st.markdown("**안전 기준** — 켜져 있는 항목만 점검해요. 확인하지 못한 항목은 '확인 불가'로 표시하고 미달로 치지 않아요.")
    s = st.session_state.safe
    for key, (lab, unit) in SAFE_LABELS.items():
        a, b = st.columns([3, 2])
        s[key]["on"] = a.checkbox(lab, value=s[key]["on"], key=f"s_on_{key}")
        if "v" in s[key] and unit:
            s[key]["v"] = b.number_input(unit, value=float(s[key]["v"]), step=1.0, key=f"s_v_{key}")
    st.divider()
    st.markdown("**종합 점수 가중치** — 근거 종류별로 얼마나 비중을 둘지예요(합계는 자동으로 100% 기준으로 계산해요). 0으로 두면 그 근거는 점수에서 빠져요. 모두 같게 두면 '그대로 평가'예요.")
    wt = st.session_state.weights
    wcols = st.columns(5)
    for i, fam in enumerate(score.FAMILIES):
        wt[fam] = wcols[i].number_input(fam, min_value=0, max_value=100, value=int(wt.get(fam, score.DEFAULT_WEIGHTS[fam])), step=5, key=f"wt_{i}")
    st.divider()
    st.markdown("**위험 한도** — 살 때 크기를 계산하는 기준이에요(종목 리포트의 진입 평가, 종목 발굴의 최대 매수)")
    rk = st.session_state.risk
    ra, rb, rc = st.columns(3)
    rk["per_trade"] = ra.number_input("한 번 틀렸을 때 총자산 대비 손실 한도(%)", min_value=0.1, max_value=10.0, value=float(rk["per_trade"]), step=0.25, key="rk_pt",
                                      help="무효가격까지 내려갔을 때 총자산의 이 비율만 잃도록 살 수량을 계산해요. 보통 1~2%를 많이 써요.")
    rk["max_pos"] = rb.number_input("한 종목 최대 비중(총자산 대비 %)", min_value=1.0, max_value=100.0, value=float(rk["max_pos"]), step=1.0, key="rk_mp")
    rk["min_ratio"] = rc.number_input("진입 손익비 기준(이 이상이면 유리)", min_value=1.0, max_value=5.0, value=float(rk["min_ratio"]), step=0.5, key="rk_mr")
    st.divider()
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("규칙 저장", type="primary"):
        ok, msg = save_settings()
        (st.success if ok else st.error)(msg)
    if c2.button("기본값으로 되돌리기"):
        reset_settings()
        st.rerun()
    st.caption("숫자를 바꾼 뒤 '규칙 저장'을 눌러야 구글 시트에 남아요. 누르지 않으면 이 화면에서만 적용되고 새로고침하면 사라져요.")

    st.divider()
    st.markdown("**사용설명서를 구글 시트에 기록**")
    st.caption("앱 주소, 시트의 탭 설명, 화면별 용도, 매일·매주·매월 작업 순서, 코드 반영 순서를 `사용설명서` 탭에 적어요. 비밀번호·TOKEN·인증키 같은 값은 적지 않아요.")
    try:
        cur_url = str(st.context.url)
    except Exception:
        cur_url = ""
    url = st.text_input("앱 주소", value=cur_url.split("?")[0], key="guide_url",
                        help="자동으로 채워지지 않으면 브라우저 주소창의 앱 주소를 붙여넣으세요.")
    if st.button("사용설명서 기록", key="guide_write"):
        store = get_store()
        try:
            import guide
        except ImportError:
            st.error("guide.py가 없어요. GitHub에 올린 뒤 앱을 다시 시작해 주세요.")
        else:
            if store is None:
                st.warning("데모 모드라서 시트에 쓸 수 없어요.")
            elif st.session_state.get("load_error"):
                st.error("구글 시트 연결이 불안정해서 기록하지 않았어요.")
            else:
                rows = guide.rows(url)
                try:
                    store.write("사용설명서", pd.DataFrame(rows[1:], columns=rows[0]))
                    st.success(f"구글 시트의 '사용설명서' 탭에 {len(rows) - 1}줄을 기록했어요.")
                except Exception as e:
                    st.error(f"기록에 실패했어요: {e}")


# 같이 올려야 하는 파일의 최소 버전. 예전 파일이 남아 있으면 오류 대신 올려야 할 파일을 알려준다.
REQUIRED_VERSIONS = {"signals": 4, "levels": 1, "judge": 1, "journal": 2, "score": 1, "explain": 1, "entry": 1, "plan": 1, "fund": 1, "lab": 4, "brief": 1}


def check_versions():
    import importlib
    old = []
    for name, need in REQUIRED_VERSIONS.items():
        try:
            mod = importlib.import_module(name)
        except ImportError:
            old.append(f"{name}.py (없어요)")
            continue
        if getattr(mod, "VERSION", 0) < need:
            old.append(f"{name}.py")
    if old:
        st.title("📈 내 투자 노트")
        st.error("GitHub에 올라간 파일 중 예전 버전이 남아 있어요: **" + ", ".join(old) + "**. 최신 파일을 올린 뒤 앱을 다시 시작(Reboot)해 주세요.")
        st.stop()


def main():
    check_versions()
    gate()
    load_tables()
    if "rules" not in st.session_state:
        st.session_state.rules = copy.deepcopy(sg.DEFAULT_RULES)
    if "safe" not in st.session_state:
        st.session_state.safe = safety.default_rules()
    if "disc_f" not in st.session_state:
        st.session_state.disc_f = copy.deepcopy(discover.DEFAULT_FILTERS)
    if "goal" not in st.session_state:
        st.session_state.goal = copy.deepcopy(perf.DEFAULT_GOAL)
    if "risk" not in st.session_state:
        st.session_state.risk = copy.deepcopy(entry.DEFAULTS)
    if "weights" not in st.session_state:
        st.session_state.weights = copy.deepcopy(score.DEFAULT_WEIGHTS)
    if "funds" not in st.session_state:
        st.session_state.funds = copy.deepcopy(DEFAULT_FUNDS)
    load_settings()
    load_watch()
    load_perf()
    load_cons_hist()
    load_journal()
    load_plans()
    load_hyp()
    load_scorelog()
    load_flowlog()
    load_fincache()
    rules = st.session_state.rules

    for k in ("rep_select", "jr_h", "jm_all", "disc_excl_held"):  # 다른 화면에 갔다 와도 선택을 유지한다
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]
    prefetch_prices()
    st.session_state["total_val"] = portfolio_value(rules)
    maybe_auto_snapshot(rules)

    st.title("📈 내 투자 노트")
    st.caption("시세는 무료 출처라 지연되거나 틀릴 수 있어요. 주문 전에는 증권사 앱의 시세를 꼭 확인하세요. 이 앱의 신호는 내 규칙에 해당하는지 알려주는 것이고, 투자 권유가 아니에요.")
    pages = {"내 자산": lambda: tab_assets(rules), "주간 브리핑": lambda: tab_brief(rules), "종합 순위": lambda: tab_rank(rules), "목표·성과": lambda: tab_perf(rules), "목적 자금": tab_fund, "종목 발굴": tab_discover, "안전 점검": tab_safety,
             "종목 리포트": lambda: tab_report(rules), "매매 계획": lambda: tab_plan(rules), "판단 기록": lambda: tab_journal(rules), "가설 실험실": tab_lab, "규칙": tab_rules}
    page = st.radio("화면", list(pages), horizontal=True, key="page", label_visibility="collapsed")
    pages[page]()  # 고른 화면만 계산해서 빠르다
    with st.sidebar:
        if st.button("시세 새로고침"):
            st.cache_data.clear()
            _price_store().clear()
            st.session_state.pop("_pos_cache", None)
            st.session_state.pop("_kk_cache", None)
            st.rerun()
        if get_store() is None:
            st.info("데모 모드")


main()
