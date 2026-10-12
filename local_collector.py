"""내 PC(한국 IP)에서 실행하는 DART 수집기.

DART가 해외 서버(Streamlit Cloud)의 접속을 막을 때를 위한 것이다. 이 PC에서 DART를 조회해서 구글 시트에 저장해 두면,
앱은 DART에 직접 접속하지 않고 시트에 있는 자료를 읽어서 안전 점검과 종목 발굴에 쓴다.

  1) 보유·관심 종목의 재무·공시 상세  -> 시트 '재무캐시' (앱이 쓰는 형식과 같다. 안전 점검·종목 리포트가 읽는다)
  2) 시가총액 상위 종목의 연간 주요 재무 -> 시트 '발굴재무' (종목 발굴 2단계가 읽는다)

사용법은 README의 'DART 수집기' 절을 보세요. collector_config.json에는 비밀번호가 들어가므로 GitHub에 올리면 안 됩니다."""
import argparse
import datetime as dt
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CACHE_COLS = ["종목코드", "갱신일", "데이터"]
DISC_COLS = ["종목코드", "재무연도", "수집일", "데이터"]
FRESH_DAYS = 7
HOLD_COLS = ["증권사", "종목코드", "종목명", "꼬리표", "수량", "평균단가", "매수일"]
WATCH_COLS = ["종목코드", "종목명"]


def load_config(path):
    if not os.path.exists(path):
        raise SystemExit(f"설정 파일이 없어요: {path}\n collector_config.example.json을 복사해서 collector_config.json으로 만들고 값을 채워 주세요.")
    cfg = json.load(open(path, encoding="utf-8"))
    miss = [k for k in ("GAS_URL", "GAS_TOKEN", "DART_API_KEY") if not str(cfg.get(k, "")).strip() or "여기에" in str(cfg.get(k))]
    if miss:
        raise SystemExit("설정 파일에 값이 비어 있어요: " + ", ".join(miss))
    return cfg


def is_fresh(d, force=False):
    if force:
        return False
    try:
        age = (dt.date.today() - dt.date.fromisoformat(d["fetched"])).days
    except Exception:
        return False
    if d.get("source") == "naver":      # 네이버 대체 자료는 DART로 바꿔 둔다
        return False
    return age <= (0 if d.get("errors") else FRESH_DAYS)


def read_codes(st, extra=()):
    codes = []
    try:
        h = st.read("보유종목", HOLD_COLS)
        codes += [str(c).strip().zfill(6) for c in h["종목코드"] if str(c).strip()]
    except Exception as e:
        print("보유종목을 읽지 못했어요:", e)
    try:
        w = st.read("관심종목", WATCH_COLS)
        codes += [str(c).strip().zfill(6) for c in w["종목코드"] if str(c).strip()]
    except Exception as e:
        print("관심종목을 읽지 못했어요:", e)
    codes += [str(c).strip().zfill(6) for c in extra if str(c).strip()]
    return list(dict.fromkeys(codes))


def collect_holdings(st, dart, codes, force=False):
    import dart_data
    cache = {}
    for r in st.read("재무캐시", CACHE_COLS).to_dict("records"):
        try:
            cache[str(r["종목코드"]).zfill(6)] = json.loads(r["데이터"])
        except Exception:
            pass
    todo = [c for c in codes if not (c in cache and is_fresh(cache[c], force))]
    print(f"[재무캐시] 대상 {len(codes)}종목 중 새로 가져올 것 {len(todo)}종목")
    ok = 0
    for i, c in enumerate(todo, 1):
        try:
            d = dart_data.fetch_all(dart, c)
        except Exception as e:
            msg = dart_data.redact(e)
            print(f"  ({i}/{len(todo)}) {c} 실패: {msg[:100]}")
            if any(k in msg for k in ("010", "011", "020")):
                print("  인증키 문제이거나 하루 사용 한도를 넘었어요. 여기서 멈춰요.")
                break
            continue
        if d.get("errors") and d.get("fin") is None:
            print(f"  ({i}/{len(todo)}) {c} 재무를 못 받았어요: {list(d['errors'].values())[:1]}")
        else:
            ok += 1
            print(f"  ({i}/{len(todo)}) {c} 완료")
        cache[c] = d
    rows = [[c, d.get("fetched", ""), json.dumps(d, ensure_ascii=False)] for c, d in cache.items()]
    st.write("재무캐시", pd.DataFrame(rows, columns=CACHE_COLS))
    print(f"[재무캐시] 저장 완료(성공 {ok}종목, 시트 전체 {len(rows)}종목)")


def collect_discover(st, dart, top_n):
    import discover
    uni = discover.load_listing()
    uni = uni[[discover.is_common_stock(c, n) for c, n in zip(uni["Code"], uni["Name"])]]
    codes = list(uni.sort_values("Marcap", ascending=False)["Code"].head(top_n))
    print(f"[발굴재무] 시가총액 상위 {len(codes)}종목의 연간 주요 재무를 조회해요")
    fin, errs = discover.bulk_financials(dart, codes, progress=lambda m: print("  " + m))
    for e in errs[:3]:
        print("  경고:", e)
    today = dt.date.today().isoformat()
    rows = [[c, rec.get("year", ""), today, json.dumps(rec, ensure_ascii=False)] for c, rec in fin.items()]
    if not rows:
        print("[발굴재무] 받은 자료가 없어서 시트를 건드리지 않았어요.")
        return
    st.write("발굴재무", pd.DataFrame(rows, columns=DISC_COLS))
    print(f"[발굴재무] 저장 완료({len(rows)}종목)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="내 PC에서 DART를 조회해 구글 시트에 저장해요")
    ap.add_argument("--config", default=os.path.join(HERE, "collector_config.json"))
    ap.add_argument("--only", choices=["holdings", "discover"], help="하나만 실행(기본은 둘 다)")
    ap.add_argument("--top", type=int, default=600, help="종목 발굴용으로 조회할 시가총액 상위 종목 수(기본 600)")
    ap.add_argument("--codes", default="", help="추가로 점검할 종목코드(쉼표로 구분)")
    ap.add_argument("--force", action="store_true", help="최근에 가져온 종목도 다시 가져와요")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    import dart_data
    import store as store_mod
    st = store_mod.GasStore(cfg["GAS_URL"], cfg["GAS_TOKEN"])
    try:
        dart = dart_data.DartClient(cfg["DART_API_KEY"])
    except Exception as e:
        raise SystemExit("DART에 연결하지 못했어요: " + dart_data.redact(e) + "\n이 PC에서도 안 된다면 DART 서버 문제일 수 있어요. 잠시 뒤 다시 해 보세요.")
    print("DART 연결 성공")
    if args.only != "discover":
        collect_holdings(st, dart, read_codes(st, args.codes.split(",")), args.force)
    if args.only != "holdings":
        collect_discover(st, dart, args.top)
    print("끝났어요. 앱을 새로고침하면 반영돼요.")


if __name__ == "__main__":
    main()
