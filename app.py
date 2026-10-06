"""내 투자 노트 - 2단계: 내 자산 + 안전 점검(DART) + 종목 리포트 (실제 시세, 구글 시트 저장)"""
import copy
import datetime as dt
import hmac
import json

import altair as alt
import pandas as pd
import streamlit as st

import dart_data
import market
import safety
import signals as sg
from store import GasStore

st.set_page_config(page_title="내 투자 노트", page_icon="📈", layout="wide")

BROKERS = ["키움", "한국투자", "카카오", "토스"]
TAGS = ["중장기", "스윙"]
HOLD_COLS = ["증권사", "종목코드", "종목명", "꼬리표", "수량", "평균단가"]
KAKAO_COLS = ["종목명", "티커", "하루금액", "시작일"]
CONC_LIMIT = 20  # 한 종목 쏠림 경고 기준(%)
CACHE_COLS = ["종목코드", "갱신일", "데이터"]
FRESH_DAYS = 7  # 재무·공시 데이터를 이 기간 안에는 다시 가져오지 않는다
STATUS_ICON = {"pass": "✅", "fail": "❌", "unknown": "❔", "na": "➖"}
SAFE_LABELS = {
    "cap": ("시가총액", "조원 이상"), "loss": ("최근 3년 영업적자", "번 이하"), "cover": ("이자보상배율", "배 이상"),
    "debt": ("부채비율", "% 이하"), "ocf": ("영업활동현금흐름 흑자", None), "impair": ("자본잠식 없음", None),
    "audit": ("감사의견 적정", None), "distress": ("부도·회생·관리절차 공시 없음", None),
    "divy": ("최근 3년 중 현금배당", "년 이상"), "major": ("최대주주 지분율", "% 이상"),
    "issue": ("최근 3년 유상증자·전환사채 등 발행", "번 이하"), "tv": ("일평균 거래대금", "억원 이상"),
}

DEMO_HOLD = pd.DataFrame(
    [
        ["키움", "005930", "삼성전자", "중장기", 10, 60000],
        ["키움", "000660", "SK하이닉스", "스윙", 3, 150000],
        ["한국투자", "005380", "현대차", "중장기", 5, 200000],
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
@st.cache_resource(show_spinner=False)
def _make_dart(key):
    return dart_data.DartClient(key)


def get_dart():
    key = secret("DART_API_KEY")
    if not key:
        return None, "DART_API_KEY가 설정되어 있지 않아요."
    try:
        return _make_dart(key), None
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


def safety_for(code):
    """캐시에 있는 DART 데이터로 안전 점검. 아직 안 가져왔으면 None."""
    d = st.session_state.get("fincache", {}).get(code)
    if d is None:
        return None
    df = hist_kr(code)
    price = float(df["Close"].iloc[-1]) if len(df) else None
    m = safety.derive(d, price, market.avg_trading_value_eok(df))
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


# ---------- 계산 ----------
def build_positions(hold, rules):
    rows = []
    for r in hold.to_dict("records"):
        df = hist_kr(r["종목코드"])
        ind = sg.indicators(df)
        price = float(df["Close"].iloc[-1]) if len(df) else float("nan")
        qty, avg = float(r["수량"]), float(r["평균단가"])
        sigs = sg.signals(ind, r["꼬리표"], avg, rules) if ind and avg > 0 else []
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
                "평가금액": qty * price, "투자원금": qty * avg,
                "수익률": (price / avg - 1) * 100 if avg > 0 else float("nan"),
                "흐름": sg.flow_of(ind)[0] if ind else "시세 부족",
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
        show = pos[["증권사", "종목명", "꼬리표", "수량", "평균단가", "현재가", "평가금액", "수익률", "흐름", "안전", "판단", "신호"]]
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            column_config={
                "수량": st.column_config.NumberColumn(format="%,d"),
                "평균단가": st.column_config.NumberColumn(format="%,d원"),
                "현재가": st.column_config.NumberColumn(format="%,d원"),
                "평가금액": st.column_config.NumberColumn(format="%,d원"),
                "수익률": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    if len(kk):
        st.subheader("카카오 소수점 정기매수")
        kk2 = kk.copy()
        kk2["손익률"] = (kk2["평가금액"] / kk2["투자원금"] - 1) * 100
        st.dataframe(
            kk2, width="stretch", hide_index=True,
            column_config={
                "투자원금": st.column_config.NumberColumn(format="%,d원"),
                "평가금액": st.column_config.NumberColumn(format="%,d원"),
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
    st.caption("안전 기준을 통과해도 안전하다는 보증은 아니에요. 투자경고·관리종목 지정 여부는 DART에서 가져올 수 없어서 이번 점검에 포함되지 않아요.")
    with st.expander("근거 데이터 보기 (증권사 앱·네이버 금융과 비교해 보세요)"):
        st.json(d)


def tab_report(rules):
    hold = st.session_state.hold
    opts = {f"{r['종목명'] or r['종목코드']} ({r['종목코드']}) · {r['증권사']}": r for r in hold.to_dict("records")}
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
    ind = sg.indicators(df)
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
    if sigs:
        for s in sigs:
            icon = {"주의": "⚠️", "매수검토": "🟢", "알림": "🔔"}[s["kind"]]
            st.markdown(f"{icon} **{s['title']}** — {s['detail']}")
        st.markdown(f"정리: **{sg.verdict(sigs)}**")
    else:
        st.write("지금은 내 기준에 걸리는 신호가 없어요.")
    report_financials(code)
    st.caption("수급·증권사 리포트는 다음 단계에서 이 화면에 붙어요.")



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
    st.caption("자세한 수치와 근거는 '종목 리포트' 탭에서 종목을 고르면 볼 수 있어요. 투자경고·관리종목 지정은 DART에서 가져올 수 없어 점검에 포함되지 않아요.")


def tab_rules():
    st.markdown("규칙 숫자를 바꾸면 신호가 바로 달라져요. 지금은 이 화면을 닫으면 기본값으로 돌아가요(규칙 저장은 다음 단계).")
    r = st.session_state.rules
    a, b = st.columns(2)
    with a:
        st.markdown("**중장기**")
        r["중장기"]["점검선"] = st.number_input("점검선(손실 %)", value=r["중장기"]["점검선"], step=1.0)
        r["중장기"]["목표"] = st.number_input("목표수익률(%)", value=r["중장기"]["목표"], step=1.0, key="l_t")
        r["중장기"]["고점권"] = st.number_input("52주 고점권 기준(%)", value=r["중장기"]["고점권"], step=1.0)
    with b:
        st.markdown("**스윙**")
        r["스윙"]["손절선"] = st.number_input("손절선(%)", value=r["스윙"]["손절선"], step=0.5)
        r["스윙"]["목표"] = st.number_input("목표수익률(%) ", value=r["스윙"]["목표"], step=1.0, key="s_t")
        r["스윙"]["RSI과열"] = st.number_input("RSI 과열(이상)", value=r["스윙"]["RSI과열"], step=1.0)
        r["스윙"]["눌림_하단"] = st.number_input("눌림 RSI 하단", value=r["스윙"]["눌림_하단"], step=1.0)
        r["스윙"]["눌림_상단"] = st.number_input("눌림 RSI 상단", value=r["스윙"]["눌림_상단"], step=1.0)
    st.divider()
    st.markdown("**안전 기준** — 켜져 있는 항목만 점검해요. 확인하지 못한 항목은 '확인 불가'로 표시하고 미달로 치지 않아요.")
    s = st.session_state.safe
    for key, (lab, unit) in SAFE_LABELS.items():
        a, b = st.columns([3, 2])
        s[key]["on"] = a.checkbox(lab, value=s[key]["on"], key=f"s_on_{key}")
        if "v" in s[key] and unit:
            s[key]["v"] = b.number_input(unit, value=float(s[key]["v"]), step=1.0, key=f"s_v_{key}")


def main():
    gate()
    load_tables()
    if "rules" not in st.session_state:
        st.session_state.rules = copy.deepcopy(sg.DEFAULT_RULES)
    if "safe" not in st.session_state:
        st.session_state.safe = safety.default_rules()
    load_fincache()
    rules = st.session_state.rules

    st.title("📈 내 투자 노트")
    st.caption("시세는 무료 출처라 지연되거나 틀릴 수 있어요. 주문 전에는 증권사 앱의 시세를 꼭 확인하세요. 이 앱의 신호는 내 규칙에 해당하는지 알려주는 것이고, 투자 권유가 아니에요.")
    t1, t2, t3, t4 = st.tabs(["내 자산", "안전 점검", "종목 리포트", "규칙"])
    with t1:
        tab_assets(rules)
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
