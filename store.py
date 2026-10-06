"""구글 시트 저장소. Apps Script 웹앱(Code.gs)을 통해 읽고 쓴다."""
import pandas as pd
import requests


class GasStore:
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token

    def _post(self, payload: dict) -> dict:
        payload = dict(payload, token=self.token)
        try:
            r = requests.post(self.url, json=payload, timeout=30)
            r.raise_for_status()
        except requests.Timeout:
            raise RuntimeError("구글 시트(Apps Script)가 제때 응답하지 않았어요(시간 초과).")
        except requests.HTTPError as e:
            raise RuntimeError(f"구글 시트(Apps Script) 응답 오류: HTTP {e.response.status_code if e.response is not None else '?'}")
        except requests.RequestException as e:
            raise RuntimeError(f"구글 시트(Apps Script)에 접속하지 못했어요({type(e).__name__}).")
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError("Apps Script 응답이 JSON이 아니에요. 배포 설정(접근 권한: 모든 사용자)을 확인해 주세요.")
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "알 수 없는 오류"))
        return data

    def read(self, sheet: str, columns: list[str]) -> pd.DataFrame:
        data = self._post({"action": "read", "sheet": sheet})
        values = data.get("values", [])
        if len(values) < 2:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(values[1:], columns=values[0])
        for c in columns:
            if c not in df.columns:
                df[c] = ""
        df = df[columns]
        return df[~(df.astype(str).apply(lambda s: s.str.strip()).eq("").all(axis=1))].reset_index(drop=True)

    def write(self, sheet: str, df: pd.DataFrame) -> None:
        values = [list(map(str, df.columns))] + df.fillna("").astype(str).values.tolist()
        self._post({"action": "write", "sheet": sheet, "values": values})
