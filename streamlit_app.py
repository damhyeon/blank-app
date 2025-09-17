# streamlit_app.py
"""
Streamlit 대시보드: '만약 내가 사는 곳이 해수면 상승으로 가라앉는다면?' (한국어 UI)
- 목적: 공식 공개데이터(글로벌·해수면·해수면 온도 등)로 먼저 대시보드 생성 후,
       프롬프트 Input에 제공된 '한국(국립해양조사원) 21개 관측소 1991-2020' 데이터를 사용해 별도 대시보드 생성.
- 데이터 출처(주요):
    NOAA Global/altimeter/tide datasets / Climate.gov / PSL: https://psl.noaa.gov/data/timeseries/month/SEALEVEL/  (NOAA PSL)
    NOAA Sea Level Rise tools / OCM: https://coast.noaa.gov/slrdata/
    Global sea level (DataHub mirror): https://datahub.io/core/sea-level-rise (raw CSV: /r/sea-level.csv)
    한국(국립해양조사원) 분석 보도자료 (1991-2020, 21개 관측소): https://www.mof.go.kr/doc/ko/selectDoc.do?docSeq=44140
    투발루 관련 연설(참고 기사): https://www.theguardian.com/environment/2021/nov/08/tuvalu-minister-to-address-cop26-knee-deep-in-seawater-to-highlight-climate-crisis
- 구현 규칙 요약:
    - 표준화: date, value, group(optional)
    - 전처리: 결측/형변환/중복 제거/미래 데이터(로컬 자정 이후) 제거
    - 캐싱: @st.cache_data 사용
    - CSV 다운로드 제공
    - API 실패 시 재시도 → 실패하면 예시 데이터로 자동 대체(화면 안내)
- 주의: 이 앱은 Codespaces/로컬 어디서나 실행 가능하도록 설계되었음.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from io import StringIO, BytesIO
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.express as px
import plotly.graph_objects as go
import pdfplumber

# ------------------------
# 설정: 한국어 UI, 페이지
# ------------------------
st.set_page_config(page_title="해수면 상승 대시보드", layout="wide")
st.title("📘 보고서: 만약 내가 사는 곳이 해수면 상승으로 가라앉는다면?")
st.caption("공개 데이터(글로벌 NOAA 등) → 한국 관측소(1991-2020, 21개) 순으로 대시보드 표시합니다. 모든 라벨은 한국어입니다.")

# Pretendard 폰트 적용 시도 (있으면 적용, 없으면 무시)
st.markdown("""
<style>
@font-face {
  font-family: 'Pretendard';
  src: url('/fonts/Pretendard-Bold.ttf') format('truetype');
  font-weight: 700;
  font-style: normal;
}
html, body, [class*="css"] { font-family: 'Pretendard', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
</style>
""", unsafe_allow_html=True)

TODAY = datetime.datetime.now().date()

# ------------------------
# HTTP 헬퍼: 재시도 로직
# ------------------------
def requests_get_retry(url, max_retries=3, timeout=15):
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=0.8, status_forcelist=[429,500,502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r

# ------------------------
# 공개 데이터: 시도적으로 가져오기
# - 1) Global sea-level (DataHub NOAA/CSRIO mirror)
# - 2) NOAA/PSL monthly sea-level timeseries (fallback)
# - 3) NOAA OISST(해수면 온도) 요약(간단)
# ------------------------
@st.cache_data(show_spinner=False)
def fetch_public_sea_level():
    # 시도 순서: DataHub raw CSV -> NOAA PSL CSV -> 실패시 None
    sources = [
        ("DataHub sea-level (CSV mirror)", "https://datahub.io/core/sea-level-rise/r/sea-level.csv"),
        ("NOAA PSL Sealevel (CSV)", "https://psl.noaa.gov/data/timeseries/month/SEALEVEL/sealevel.monthly.mean.csv"),
        ("NOAA climate.gov derived (CSV)", "https://www.climate.gov/sites/default/files/global_mean_sea_level.csv")
    ]
    last_err = None
    for name, url in sources:
        try:
            r = requests_get_retry(url)
            text = r.text
            return {"text": text, "url": url, "name": name}
        except Exception as e:
            last_err = e
            continue
    return None

@st.cache_data(show_spinner=False)
def fetch_noaa_sst_sample():
    # NOAA OISST 페이지는 그리드인데 여기선 간단히 메타정보만 가져오고, 실패시 None
    url = "https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html"
    try:
        r = requests_get_retry(url)
        return {"text": r.text, "url": url}
    except:
        return None

public_sea = fetch_public_sea_level()
public_sst = fetch_noaa_sst_sample()

# ------------------------
# 공개 데이터 처리 (표준화)
# 기대: DataHub sea-level CSV 형태 -> columns: 'Year','Month','GMSL' 또는 date-like
# ------------------------
def parse_sea_csv(text_blob):
    # 다양한 포맷 시도를 해서 date,value 표준형 반환
    try:
        df = pd.read_csv(StringIO(text_blob))
    except Exception:
        # 시도: skip bad lines
        df = pd.read_csv(StringIO(text_blob), error_bad_lines=False)
    # 후보 컬럼 탐색
    cols = [c.lower() for c in df.columns]
    # 케이스별 처리
    # 1) 'Time' or 'time' + 'GMSL' or 'gmsl'
    if any('time' in c for c in cols) and any('gmsl' in c for c in cols):
        # find columns
        time_col = [c for c in df.columns if 'time' in c.lower()][0]
        val_col = [c for c in df.columns if 'gmsl' in c.lower()][0]
        df2 = df[[time_col, val_col]].rename(columns={time_col:'date', val_col:'value'})
        # try parse date
        df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
        df2['value'] = pd.to_numeric(df2['value'], errors='coerce')
        return df2.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    # 2) 월별 표가 Year, Jan..Dec -> convert to long
    month_cols = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    if 'Year' in df.columns or 'year' in cols:
        year_col = [c for c in df.columns if c.lower()=='year'][0] if 'Year' in df.columns or 'year' in cols else df.columns[0]
        available_months = [m for m in month_cols if m in df.columns]
        records = []
        for _, row in df.iterrows():
            try:
                year = int(row[year_col])
            except:
                continue
            for i, mon in enumerate(available_months, start=1):
                raw = row.get(mon, np.nan)
                try:
                    val = float(raw)
                except:
                    val = np.nan
                try:
                    date = pd.to_datetime(f"{year}-{i:02d}-01")
                except:
                    continue
                records.append({'date': date, 'value': val})
        if records:
            df_long = pd.DataFrame.from_records(records)
            return df_long.sort_values('date').reset_index(drop=True)
    # 3) Fall back: try to find first two columns: date,value
    if df.shape[1] >= 2:
        df2 = df.iloc[:, :2].copy()
        df2.columns = ['date','value']
        df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
        df2['value'] = pd.to_numeric(df2['value'], errors='coerce')
        return df2.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return pd.DataFrame(columns=['date','value'])

# 공개 데이터 파싱
using_public_example = False
if public_sea is None:
    using_public_example = True
    st.warning("공개 데이터(글로벌 해수면) 다운로드 실패 — 예시 데이터로 대체합니다.")
    # 간단 예시: 연도별 누적(mm)
    example = """date,value
1993-01-01,0.0
1995-01-01,5.2
2000-01-01,10.8
2005-01-01,18.0
2010-01-01,34.5
2015-01-01,45.1
2020-01-01,60.2
2023-01-01,72.4
"""
    public_df = parse_sea_csv(example)
else:
    public_df = parse_sea_csv(public_sea['text'])
    # remove future dates (오늘 로컬 자정 이후 데이터 제거)
    public_df = public_df[public_df['date'].dt.date <= TODAY].reset_index(drop=True)
    if public_df.empty:
        using_public_example = True
        st.warning("공개 데이터 파싱 후 유효한 시계열이 없어 예시 데이터로 대체합니다.")
        example = """date,value
1993-01-01,0.0
1995-01-01,5.2
2000-01-01,10.8
2005-01-01,18.0
2010-01-01,34.5
2015-01-01,45.1
2020-01-01,60.2
2023-01-01,72.4
"""
        public_df = parse_sea_csv(example)

# 전처리: 결측 보간/중복 제거/정렬
def preprocess_standard_ts(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if df['value'].isna().any():
        df['value'] = df['value'].interpolate(method='time', limit_direction='both')
    # remove future dates (again, safety)
    df = df[df['date'].dt.date <= TODAY].reset_index(drop=True)
    return df

public_df = preprocess_standard_ts(public_df)

# ------------------------
# 한국(국립해양조사원) 21개 관측소 데이터 시도 가져오기 (MOF 보도자료 PDF)
# - 출처: https://www.mof.go.kr/doc/ko/selectDoc.do?docSeq=44140
# - 방법: PDF 내 표 추출 시도(poor man's approach). 실패 시 예시 데이터 사용.
# ------------------------
@st.cache_data(show_spinner=False)
def fetch_korean_mof_pdf():
    pdf_page = "https://www.mof.go.kr/jfile/readDownloadFile.do?fileNum=1&fileType=MOF_ARTICLE&fileTypeSeq=44140"
    try:
        r = requests_get_retry(pdf_page)
        return {"bytes": r.content, "url": pdf_page}
    except Exception:
        return None

korea_pdf = fetch_korean_mof_pdf()
using_korea_example = False

def parse_mof_pdf_to_df(pdf_bytes):
    # 시도적으로 PDF에서 '21개' 관련 표를 추출. (환경에 따라 실패할 수 있음)
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            all_tables = []
            for p in pdf.pages:
                tables = p.extract_tables()
                for t in tables:
                    # 테이블은 리스트 형태. 합치기 위해 DF로 변환 시도.
                    df_t = pd.DataFrame(t[1:], columns=t[0]) if len(t) >=2 else None
                    if df_t is not None:
                        all_tables.append(df_t)
            if not all_tables:
                return None
            combined = pd.concat(all_tables, ignore_index=True)
            # 단순히 리턴 (후속에서 표준화 시도)
            return combined
    except Exception:
        return None

korea_df_raw = None
if korea_pdf:
    korea_df_raw = parse_mof_pdf_to_df(korea_pdf['bytes'])
if korea_df_raw is None:
    using_korea_example = True
    st.info("한국 관측소(1991-2020) 데이터: 원문(PDF)에서 표 추출을 시도했으나 실패하거나 표 형식이 다양하여 내장 예시 데이터로 보여드립니다.")
    # 예시: 21개 관측소 중 일부를 샘플로 만든 데이터 (연-월-관측소-상승률(mm))
    korea_example = """date,station,value,region
1991-01-01,울릉도,2.3,동해
1995-01-01,포항,2.9,동해
2000-01-01,보령,3.1,서해
2005-01-01,인천,3.4,서해
2010-01-01,속초,3.8,동해
2015-01-01,울릉도,4.9,동해
2020-01-01,포항,6.17,동해
"""
    korea_df = pd.read_csv(StringIO(korea_example))
else:
    # 단순 표준화 시도: 컬럼명이 포함되어 있으면 'date' 또는 '연도'등으로 변환
    dfk = korea_df_raw.copy()
    # 시도적으로 연도/월/값 컬럼 찾기
    cols = [c.strip() for c in dfk.columns]
    lower = [c.lower() for c in cols]
    # find year/month or date
    date_col = None
    if any('date' in c for c in lower):
        date_col = cols[lower.index([c for c in lower if 'date' in c][0])]
    elif any('연' in c or 'year' in c for c in lower):
        date_col = cols[0]
    # find value-like column
    value_col = None
    for candidate in ['값','해수면','상승','mm','rate','value']:
        matches = [c for c in lower if candidate in c]
        if matches:
            value_col = cols[lower.index(matches[0])]
            break
    # station
    station_col = None
    for candidate in ['관측','station','지점','관측소','site','location']:
        matches = [c for c in lower if candidate in c]
        if matches:
            station_col = cols[lower.index(matches[0])]
            break
    # try build minimal df
