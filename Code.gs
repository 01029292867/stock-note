/**
 * 내 투자 노트 - 구글 시트 저장소 (Apps Script)
 * 1) 구글 시트를 새로 만들고  확장 프로그램 > Apps Script 에 이 코드를 붙여넣기
 * 2) 프로젝트 설정 > 스크립트 속성 에 TOKEN 이라는 이름으로 긴 비밀 문자열을 추가
 * 3) 배포 > 새 배포 > 유형: 웹 앱 > 실행: 나 / 액세스: 모든 사용자 > 배포 후 웹 앱 URL 복사
 */
function doPost(e) {
  try {
    var req = JSON.parse(e.postData.contents);
    var token = PropertiesService.getScriptProperties().getProperty('TOKEN');
    if (!token || req.token !== token) return out_({ ok: false, error: 'unauthorized' });
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var name = String(req.sheet || '');
    if (!name) return out_({ ok: false, error: 'sheet required' });
    var sh = ss.getSheetByName(name) || ss.insertSheet(name);

    if (req.action === 'read') {
      var values = sh.getLastRow() === 0 ? [] : sh.getDataRange().getValues();
      values = values.map(function (row) {
        return row.map(function (v) { return v instanceof Date ? Utilities.formatDate(v, 'Asia/Seoul', 'yyyy-MM-dd') : String(v); });
      });
      return out_({ ok: true, values: values });
    }
    if (req.action === 'write') {
      var vals = req.values || [];
      sh.clear();
      if (vals.length > 0 && vals[0].length > 0) {
        var rng = sh.getRange(1, 1, vals.length, vals[0].length);
        rng.setNumberFormat('@');
        rng.setValues(vals);
      }
      return out_({ ok: true });
    }
    return out_({ ok: false, error: 'unknown action' });
  } catch (err) {
    return out_({ ok: false, error: String(err) });
  }
}
function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
