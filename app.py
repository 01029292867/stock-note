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
import dart_data
import discover
import flows
import journal
import judge
import market
import perf
import reports
import safety
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
        saved = {r["이름"]: json.loads(r["값"]) for r in df.to_dict("records") if r.get("이름") in ("rules", "safe", "disc", "goal")}
    except Exception:
        return  # 설정을 못 읽으면 기본값으로 시작한다(저장된 값을 지우지는 않는다)
    _merge(st.session_state.rules, saved.get("rules"))
    _merge(st.session_state.safe, saved.get("safe"))
    _merge(st.session_state.disc_f, saved.get("disc"))
    _merge(st.session_state.goal, saved.get("goal"))


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
                ["goal", json.dumps(st.session_state.goal, ensure_ascii=False)]]
        store.write("설정", pd.DataFrame(rows, columns=SETTINGS_COLS))
        return True, "규칙을 저장했어요. 이제 새로고침하거나 폰에서 열어도 그대로예요."
    except Exception as e:
        return False, f"저장에 실패했어요: {e}"


def reset_settings():
    st.session_state.rules = copy.deepcopy(sg.DEFAULT_RULES)
    st.session_state.safe = safety.default_rules()
    st.session_state.disc_f = copy.deepcopy(discover.DEFAULT_FILTERS)
    st.session_state.goal = copy.deepcopy(perf.DEFAULT_GOAL)
    for k in list(st.session_state.keys()):
        if k.startswith(("s_on_", "s_v_", "d_on_", "d_v_", "g_")) or k in ("l_t", "s_t", "d_maxn", "l_ta", "l_tw", "s_ta", "s_tw", "l_td", "s_td", "l_sp", "s_sp"):
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
    for c in dict.fromkeys(list(st.session_state.hold["종목코드"]) + list(st.session_state.get("watch", pd.DataFrame(columns=WATCH_COLS))["종목코드"])):
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
    key = _sig(hold.to_json(), json.dumps(rules, sort_keys=True), json.dumps(ss.safe, sort_keys=True),
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
        show = pos[["증권사", "종목명", "꼬리표", "수량", "평균단가", "현재가", "평가금액", "수익금액", "수익률", "고점 대비(%)", "수익 보호선(원)", "보호선 상태", "지지선(원)", "지지선까지(%)", "평소 되돌림(%)", "흐름", "수급(5일)", "목표가 여력(%)", "안전", "판단", "신호"]]
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





def record_form(card, code, name, key, default_decision="보유 유지"):
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
        ok, msg = save_journal([make_record(card, code, name, dec, reason, conf, inv, memo)])
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()


def make_record(card, code, name, decision, reason, conf, invalid, memo):
    base = card["base"]
    return {"id": dt.datetime.now().strftime("%Y%m%d%H%M%S") + code, "날짜": dt.date.today().isoformat(), "종목코드": code, "종목명": name,
            "결정": decision, "이유": reason, "확신도": conf, "기록가": card["price"], "평단": card["avg"], "수량": card["qty"], "꼬리표": card["tag"],
            "상승여력": card["up"], "하락위험": base["pct"] if base else None, "손익비": card["ratio"], "등급": card["grade"],
            "필요승률": card["need_win"], "무효가격": invalid or None, "메모": memo}


def render_card(card, name):
    g = card["grade"]
    st.subheader(f"종합 판단 카드 — {name}")
    k = st.columns(5)
    k[0].metric("보수적 상승 여력", "-" if card["up"] is None else f"+{card['up']:.1f}%")
    k[1].metric("하락 위험(중기)", "-" if not card["base"] else f"{card['base']['pct']:.1f}%")
    k[2].metric("손익비", "-" if card["ratio"] is None else f"{card['ratio']:.2f}", g, delta_color="off")
    k[3].metric("필요 승률", "-" if card["need_win"] is None else f"{card['need_win']:.0f}%", help="이 손익비에서 본전이 되려면 이 정도 이상 맞혀야 해요. 51%보다 높으면 구조적으로 불리해요.")
    k[4].metric("근거 충족도", f"{card['n_have']}/4", " · ".join(n for n, v in card["have"].items() if not v) or "모두 있음", delta_color="off")
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

    st.subheader(f"내 기준으로 보면 ({tag})")
    sigs = sg.signals(ind, tag, avg, rules)
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
    else:
        st.write("지금은 내 기준에 걸리는 신호가 없어요.")
    card = card_for(code, tag, avg, float(held["수량"]) if held else 0.0, (held or {}).get("매수일") or None)
    if card:
        render_card(card, (held or {}).get("종목명") or code)
        with st.expander("이 카드로 판단 기록하기"):
            record_form(card, code, (held or {}).get("종목명") or code, f"rf_{code}", "보유 유지" if held else "관망(사지 않음)")
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
    st.checkbox("이미 보유한 종목은 제외", value=True, key="disc_excl_held")
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
        "목표가 여력(%)": st.column_config.NumberColumn(format="%.0f")})
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
        show_all = st.checkbox("점검 대상이 아닌 종목도 보기", value=False, key="jm_all")
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
            rows = [make_record(c, code, ag["name"], dec, "월간 점검", conf, c.get("invalid"), memo) for code, (c, ag, dec, conf, memo) in pick.items()]
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
    h = st.radio("몇 거래일 뒤 결과로 볼까요", list(journal.HORIZONS), index=1, horizontal=True, format_func=lambda x: f"{x}일(약 {x // 20}개월)", key="jr_h")
    done = int(ev[f"v{h}"].notna().sum())
    st.caption(f"기록 {len(ev)}건 중 {h}일이 지나 결과가 나온 것은 {done}건이에요. 방향은 이렇게 봐요: 사거나 들고 있는 결정은 이후 오르면 맞은 판단, 팔거나 사지 않은 결정은 이후 내리면 맞은 판단이에요(손익은 그 방향으로 계산).")
    if done:
        for title, by in (("결정 유형별", None), ("확신도별", "확신구분"), ("기록 당시 카드 등급별", "등급"), ("이유별", "이유")):
            e2 = ev.copy()
            e2["확신구분"] = e2["확신도"].map(journal.confidence_bucket)
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
REQUIRED_VERSIONS = {"signals": 4, "levels": 1, "judge": 1, "journal": 1}


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
    load_settings()
    load_watch()
    load_perf()
    load_cons_hist()
    load_journal()
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
    pages = {"내 자산": lambda: tab_assets(rules), "목표·성과": lambda: tab_perf(rules), "종목 발굴": tab_discover, "안전 점검": tab_safety,
             "종목 리포트": lambda: tab_report(rules), "판단 기록": lambda: tab_journal(rules), "규칙": tab_rules}
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
