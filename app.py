"""내 투자 노트 - 2단계: 내 자산 + 안전 점검(DART) + 종목 리포트 (실제 시세, 구글 시트 저장)"""
import copy
import datetime as dt
import hmac
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import consensus
import dart_data
import discover
import flows
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
KAKAO_COLS = ["종목명", "티커", "하루금액", "시작일"]
CONC_LIMIT = 20  # 한 종목 쏠림 경고 기준(%)
CACHE_COLS = ["종목코드", "갱신일", "데이터"]
SETTINGS_COLS = ["이름", "값"]
WATCH_COLS = ["종목코드", "종목명"]
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
DEMO_KAKAO = pd.DataFrame([["엔비디아", "NVDA", 3000, "2026-03-02"]], columns=KAKAO_COLS)


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
        if k.startswith(("s_on_", "s_v_", "d_on_", "d_v_", "g_")) or k in ("l_t", "s_t", "d_maxn", "l_ta", "l_tw", "s_ta", "s_tw"):
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
    df["시작일"] = df["시작일"].astype(str).str.slice(0, 10)
    df["종목명"] = df["종목명"].fillna("").astype(str)
    return df.reset_index(drop=True)


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
DART_CLIENT_VER = "4"  # DartClient 코드를 바꾸면 이 숫자를 올려서 예전 객체가 재사용되지 않게 한다


@st.cache_resource(show_spinner=False)
def _make_dart(key, ver):
    return dart_data.DartClient(key)


def get_dart():
    key = secret("DART_API_KEY")
    if not key:
        return None, "DART_API_KEY가 설정되어 있지 않아요."
    try:
        return _make_dart(key, DART_CLIENT_VER), None
    except Exception as e:
        return None, f"DART 연결에 실패했어요: {e}"


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
        return None, str(e)
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
@st.cache_data(ttl=900, show_spinner=False)
def hist_kr(code):
    return market.history_kr(code)


@st.cache_data(ttl=900, show_spinner=False)
def hist_us(ticker):
    return market.history_us(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def hist_fx():
    return market.history_fx()


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
                "고점 대비(%)": prot["dd"] if prot else float("nan"),
                "수익 보호선(원)": prot["line"] if prot and prot["active"] else float("nan"),
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
        res = market.kakao_value(hist_us(r["티커"]), fx, float(r["하루금액"]), r["시작일"])
        if res is None:
            rows.append({"증권사": "카카오", "종목명": r["종목명"] or r["티커"], "투자원금": float("nan"), "평가금액": float("nan"), "회차": 0})
        else:
            cost, value, n = res
            rows.append({"증권사": "카카오", "종목명": r["종목명"] or r["티커"], "투자원금": cost, "평가금액": value, "회차": n})
    return pd.DataFrame(rows)


def man(n):
    return "-" if pd.isna(n) else f"{n / 1e4:,.0f}만원"


def pct(n):
    return "-" if pd.isna(n) else f"{n:+.1f}%"


# ---------- 화면 ----------
def tab_assets(rules):
    hold, kakao = st.session_state.hold, st.session_state.kakao
    pos = build_positions(hold, rules) if len(hold) else pd.DataFrame()
    kk = build_kakao(kakao) if len(kakao) else pd.DataFrame()

    ok_pos = pos[pos["평가금액"].notna()] if len(pos) else pos
    ok_kk = kk[kk["평가금액"].notna()] if len(kk) else kk
    val = (ok_pos["평가금액"].sum() if len(ok_pos) else 0) + (ok_kk["평가금액"].sum() if len(ok_kk) else 0)
    cost = (ok_pos["투자원금"].sum() if len(ok_pos) else 0) + (ok_kk["투자원금"].sum() if len(ok_kk) else 0)

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
        show = pos[["증권사", "종목명", "꼬리표", "수량", "평균단가", "현재가", "평가금액", "수익금액", "수익률", "고점 대비(%)", "수익 보호선(원)", "흐름", "수급(5일)", "목표가 여력(%)", "안전", "판단", "신호"]]
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
                "수익 보호선(원)": st.column_config.NumberColumn(format="%,d", help="고점 수익률이 활성 기준을 넘은 종목만 나와요. 현재가가 이 값 이하로 내려가면 수익 보호선 도달이에요."),
                "목표가 여력(%)": st.column_config.NumberColumn(format="%.0f"),
            },
        )
    if len(kk):
        st.subheader("카카오 소수점 정기매수")
        kk2 = kk.copy()
        kk2["수익금액"] = kk2["평가금액"] - kk2["투자원금"]
        kk2["손익률"] = (kk2["평가금액"] / kk2["투자원금"] - 1) * 100
        kk2 = kk2[["증권사", "종목명", "회차", "투자원금", "평가금액", "수익금액", "손익률"]]
        st.dataframe(
            kk2, width="stretch", hide_index=True,
            column_config={
                "투자원금": st.column_config.NumberColumn(format="%,d원"),
                "평가금액": st.column_config.NumberColumn(format="%,d원"),
                "수익금액": st.column_config.NumberColumn(format="%,d원"),
                "손익률": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
        st.caption("매 미국 거래일 하루 금액만큼 샀다고 보고 그날 주가·환율로 계산한 추정치예요. 실제 체결 내역과 조금 다를 수 있어요.")

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
    st.markdown("**카카오 소수점 정기매수** — 티커는 미국 종목 약어예요(예: NVDA, AAPL). 시작일은 2026-03-02 형식이에요.")
    ek = st.data_editor(
        st.session_state.kakao, num_rows="dynamic", width="stretch", hide_index=True, key="kakao_editor",
        column_config={"하루금액": st.column_config.NumberColumn(min_value=0, step=100)},
    )
    if st.button("저장", type="primary"):
        h, k = clean_hold(ed), clean_kakao(ek)
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
        st.info(f"{err} Streamlit Secrets에 DART_API_KEY를 넣으면 재무와 안전 점검이 켜져요.")
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
    pick = st.selectbox("종목 선택", labels)
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
            if prot["active"]:
                st.write(f"수익 보호선: 고점 {prot['peak']:,.0f}원({prot['peak_date']}, {prot['basis']}) − {prot['width']:.0f}% = **{prot['line']:,.0f}원** · 지금은 고점 대비 {prot['dd']:.1f}%")
            else:
                st.caption(f"수익 보호선은 아직 꺼져 있어요. 고점 기준 수익률이 +{prot['act']:.0f}%를 넘으면 켜져요(지금 고점 기준 {prot['peak_ret']:+.0f}%).")
    if sigs:
        for s in sigs:
            icon = {"주의": "⚠️", "매수검토": "🟢", "알림": "🔔"}[s["kind"]]
            st.markdown(f"{icon} **{s['title']}** — {s['detail']}")
        st.markdown(f"정리: **{sg.verdict(sigs)}**")
    else:
        st.write("지금은 내 기준에 걸리는 신호가 없어요.")
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
        st.warning(f"{err} Streamlit Secrets에 DART_API_KEY를 넣어 주세요.")
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
    pos = build_positions(hold, rules) if len(hold) else pd.DataFrame()
    kk = build_kakao(kakao) if len(kakao) else pd.DataFrame()
    missing = (pos[pos["평가금액"].isna()]["종목명"].tolist() if len(pos) else []) + (kk[kk["평가금액"].isna()]["종목명"].tolist() if len(kk) else [])
    if missing:
        return None, missing
    val = (pos["평가금액"].sum() if len(pos) else 0.0) + (kk["평가금액"].sum() if len(kk) else 0.0)
    return float(val), []


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

    # 하루 한 번 자동 기록
    if (stock_val and stock_val > 0 and st.session_state.get("perf_auto") != str(today.date())
            and not st.session_state.get("perf_err") and not st.session_state.get("load_error")
            and not (len(snaps) and (snaps["날짜"] == today).any())):
        st.session_state.perf_auto = str(today.date())
        row = pd.DataFrame([[str(today.date()), stock_val, last_cash, stock_val + last_cash, "자동", "현금은 마지막 기록값"]], columns=perf.SNAP_COLS)
        ok, msg = save_perf(snaps=pd.concat([snaps_view(snaps_raw), row], ignore_index=True))
        if ok:
            snaps_raw = st.session_state.perf_snaps
            snaps = perf.clean_snaps(snaps_raw)
        else:
            st.error(f"오늘 자산을 자동으로 기록하지 못했어요: {msg}")
    if missing:
        st.warning("시세를 가져오지 못한 종목이 있어서 오늘 자산을 자동 기록하지 않았어요: " + ", ".join(missing))

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
        st.warning(f"{err} Streamlit Secrets에 DART_API_KEY를 넣어 주세요.")
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
    with b:
        st.markdown("**스윙**")
        r["스윙"]["손절선"] = st.number_input("손절선(%)", value=r["스윙"]["손절선"], step=0.5)
        r["스윙"]["목표"] = st.number_input("목표수익률(%) ", value=r["스윙"]["목표"], step=1.0, key="s_t")
        r["스윙"]["RSI과열"] = st.number_input("RSI 과열(이상)", value=r["스윙"]["RSI과열"], step=1.0)
        r["스윙"]["눌림_하단"] = st.number_input("눌림 RSI 하단", value=r["스윙"]["눌림_하단"], step=1.0)
        r["스윙"]["눌림_상단"] = st.number_input("눌림 RSI 상단", value=r["스윙"]["눌림_상단"], step=1.0)
        r["스윙"]["보호활성"] = st.number_input("수익 보호 시작(고점 기준 수익률 % 이상) ", value=float(r["스윙"]["보호활성"]), step=1.0, key="s_ta")
        r["스윙"]["보호폭"] = st.number_input("수익 보호폭(고점 대비 % 하락) ", value=float(r["스윙"]["보호폭"]), step=1.0, key="s_tw")
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


def main():
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
    load_fincache()
    rules = st.session_state.rules

    st.title("📈 내 투자 노트")
    st.caption("시세는 무료 출처라 지연되거나 틀릴 수 있어요. 주문 전에는 증권사 앱의 시세를 꼭 확인하세요. 이 앱의 신호는 내 규칙에 해당하는지 알려주는 것이고, 투자 권유가 아니에요.")
    t1, t6, t5, t2, t3, t4 = st.tabs(["내 자산", "목표·성과", "종목 발굴", "안전 점검", "종목 리포트", "규칙"])
    with t1:
        tab_assets(rules)
    with t6:
        tab_perf(rules)
    with t5:
        tab_discover()
    with t2:
        tab_safety()
    with t3:
        tab_report(rules)
    with t4:
        tab_rules()
    with st.sidebar:
        if st.button("시세 새로고침"):
            st.cache_data.clear()
            st.rerun()
        if get_store() is None:
            st.info("데모 모드")


main()
