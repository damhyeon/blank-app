# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 공개 데이터 대시보드: NASA GISS GISTEMP 월별 전지구 온도 이상치(Anomaly) CSV 사용
    출처: https://data.giss.nasa.gov/gistemp/ 
    직접 CSV 파일: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
- 사용자 입력 대시보드: 이 프롬프트의 'Input' 섹션에 데이터가 없을 경우 예시 데이터를 사용하여 동작 시연
- 구현 규칙:
    - date, value, group(optional) 표준화
    - 결측/형변환/중복 처리
    - 미래 날짜 제거 (로컬 시스템의 현재 날짜 기준)
    - @st.cache_data 사용
    - 전처리된 표 CSV 다운로드 제공
- 비고: kaggle API 사용 시 별도 인증 안내를 주석으로 추가 (본 코드에서는 kaggle 사용하지 않음)
"""

from io import StringIO
import requests
import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dateutil import parser as dateparse
from requests.adapters import HTTPAdapter, Retry

# -------------------------
# 기본 설정 (한국어 UI)
# -------------------------
st.set_page_config(page_title="데이터 대시보드 (Streamlit + Codespaces)", layout="wide")

# Pretendard 폰트 적용 시도 (있으면 적용, 없으면 무시)
st.markdown(
    """
    <style>
    @font-face {
        font-family: 'Pretendard';
        src: url('/fonts/Pretendard-Bold.ttf') format('truetype');
        font-weight: 700;
        font-style: normal;
    }
    html, body, [class*="css"]  {
        font-family: 'Pretendard', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 데이터 대시보드 — 공개 데이터 + 사용자 입력 데이터 (한국어 UI)")
st.caption("공개 데이터: NASA GISS GISTEMP (전지구 월별 온도 이상치). 사용자 입력 데이터: 프롬프트 Input 섹션 기반 (없으면 예시 사용).")

# 유틸: 현재 로컬 날짜(앱이 실행되는 머신 시간)
TODAY = datetime.datetime.now().date()

# -------------------------
# 헬퍼 함수: 안전한 HTTP 가져오기 (재시도 + 대체 데이터)
# -------------------------
def requests_get_with_retry(url, max_retries=3, backoff=1.0, timeout=15):
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=backoff, status_forcelist=[429,500,502,503,504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp

@st.cache_data(show_spinner=False)
def fetch_gistemp_csv(url="https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"):
    """
    NASA GISTEMP CSV를 시도해 가져온다.
    실패 시 None 반환 (호출부에서 예시 데이터로 자동 대체)
    출처 주석: https://data.giss.nasa.gov/gistemp/ , CSV: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    """
    try:
        resp = requests_get_with_retry(url)
        resp.encoding = 'utf-8'
        text = resp.text
        return text
    except Exception as e:
        # 재시도 이미 수행. 여기서 실패하면 None 반환
        return None

# -------------------------
# 공개 데이터 대시보드: NASA GISTEMP 가져오기 -> 전처리
# -------------------------
st.header("1. 공개 데이터 대시보드 — NASA GISS (GISTEMP)")

with st.expander("데이터 원본 / 처리설명 (클릭해서 보기)", expanded=False):
    st.markdown("""
    - 데이터 출처: NASA GISS GISTEMP (월별 전지구 평균 온도 이상치)
      - 메인 페이지: https://data.giss.nasa.gov/gistemp/
      - CSV 파일(공식): https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    - 본 앱은 CSV 직접 가져와 전처리 (결측/중복/형변환/미래 날짜 제거) 후 시각화합니다.
    - 만약 원격 CSV 호출 실패 시 자동으로 예시(대체) 데이터로 전환하며, 화면에 안내 메시지를 표시합니다.
    - (참고) kaggle API 사용법이 필요한 경우 별도 안내 필요합니다. 이 샘플은 kaggle을 사용하지 않습니다.
    """)

raw_text = fetch_gistemp_csv()
using_example_public = False

if raw_text is None:
    using_example_public = True
    st.warning("공개 데이터(GISTEMP) 다운로드에 실패했습니다. 예시 데이터로 대체하여 표시합니다. (원인: 네트워크 또는 원격 서버 차단)")
    # 간단 예시 데이터 (연-월, anomaly)
    example_public_csv = """Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,Annual
    2020,0.92,0.79,0.98,0.91,0.85,0.84,0.90,0.86,0.76,0.84,0.89,0.95,0.86
    2021,0.98,0.76,0.85,1.02,0.92,0.88,0.95,0.90,0.80,0.88,0.94,0.99,0.90
    2022,1.05,0.92,1.10,1.12,1.03,0.98,1.00,0.99,0.88,0.96,1.02,1.08,1.03
    2023,1.12,1.00,1.15,1.20,1.10,1.05,1.08,1.02,0.95,1.03,1.09,1.14,1.08
    """
    raw_text = example_public_csv

# 파싱: NASA CSV 파일은 년도 행 + 12개월 열 + 연평균 열 구조
def parse_gistemp_table(text):
    """
    GISTEMP의 tabledata_v4 CSV/텍스트 포맷을 파싱하여
    date,value (월별) 표준형으로 반환.
    """
    # 일부 GISTEMP 파일은 헤더 또는 주석 라인으로 시작. Pandas로 읽어보고, 해를 년도 컬럼으로 사용.
    try:
        df = pd.read_csv(StringIO(text), skiprows=0)
    except Exception:
        # fallback: try different encoding/sep
        df = pd.read_csv(StringIO(text))
    # Expect columns like 'Year','Jan','Feb',...,'Dec', 'Annual'
    cols = df.columns.tolist()
    # normalize column names
    df.columns = [c.strip() for c in cols]
    month_cols = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    available_months = [m for m in month_cols if m in df.columns]
    records = []
    for _, row in df.iterrows():
        year = int(row['Year'])
        for i, mon in enumerate(available_months, start=1):
            raw_val = row.get(mon, np.nan)
            try:
                val = float(raw_val)
            except Exception:
                val = np.nan
            # create date
            try:
                date = datetime.date(year, i, 1)
            except Exception:
                continue
            records.append({'date': pd.to_datetime(date), 'value': val})
    df_long = pd.DataFrame.from_records(records)
    # 정렬, 중복 제거
    df_long = df_long.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    return df_long

public_df = parse_gistemp_table(raw_text)

# 전처리: 결측 처리(보간), 형변환, 미래 날짜 제거
def preprocess_timeseries(df):
    df = df.copy()
    # ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    # remove future dates (strictly greater than TODAY)
    df = df[df['date'].dt.date <= TODAY]
    # sort
    df = df.sort_values('date').reset_index(drop=True)
    # duplicate removal
    df = df.drop_duplicates(subset=['date'])
    # ensure numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    # simple interpolation for missing values (linear)
    if df['value'].isna().any():
        df['value'] = df['value'].interpolate(method='time', limit_direction='both')
    return df

public_df = preprocess_timeseries(public_df)

# 다운로드 버튼 제공을 위한 CSV
public_csv_bytes = public_df.to_csv(index=False).encode('utf-8')

# 공개 데이터 시각화 UI (사이드바 컨트롤)
st.subheader("공개 데이터: 전지구 월별 온도 이상치 (GISTEMP)")
with st.sidebar.expander("공개 데이터 설정 (GISTEMP)", expanded=True):
    st.write("데이터 소스: NASA GISS GISTEMP (월별)")
    show_smoothing = st.checkbox("이동평균 적용 (3개월)", value=True)
    avg_window = st.number_input("이동평균 기간 (개월)", min_value=1, max_value=24, value=3, step=1)
    chart_kind = st.radio("차트 종류", options=["꺾은선그래프", "면적그래프", "바 차트"], index=0)
    include_annual = st.checkbox("연평균(연도별) 라인 표시", value=False)

# 기본 차트 (Plotly)
if public_df.empty:
    st.error("공개 데이터가 비어있습니다.")
else:
    df_plot = public_df.copy()
    if show_smoothing and avg_window > 1:
        df_plot['smoothed'] = df_plot['value'].rolling(window=avg_window, min_periods=1, center=True).mean()
        y_col = 'smoothed'
    else:
        y_col = 'value'

    fig = go.Figure()
    if chart_kind == "꺾은선그래프":
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[y_col], mode='lines+markers', name='월별 이상치'))
    elif chart_kind == "면적그래프":
        fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[y_col], mode='lines', fill='tozeroy', name='월별 이상치(면적)'))
    else:
        fig = px.bar(df_plot, x='date', y=y_col, labels={'date':'날짜','value':'이상치 (°C)'})
    # 연평균 라인 계산 및 표시 (옵션)
    if include_annual:
        df_plot['year'] = df_plot['date'].dt.year
        annual = df_plot.groupby('year')[y_col].mean().reset_index()
        # convert year to date at mid-year for plotting
        annual['date'] = pd.to_datetime(annual['year'].astype(str) + '-07-01')
        fig.add_trace(go.Scatter(x=annual['date'], y=annual[y_col], mode='lines+markers', name='연평균', line=dict(dash='dash', width=2)))
    fig.update_layout(title="전지구 월별 온도 이상치 (NASA GISTEMP)",
                      xaxis_title="날짜",
                      yaxis_title="이상치 (°C)",
                      hovermode='x unified',
                      legend_title_text='항목')
    st.plotly_chart(fig, use_container_width=True)

# 공개 데이터 테이블 + CSV 다운로드
with st.expander("전처리된 공개 데이터 표 / CSV 다운로드", expanded=False):
    st.dataframe(public_df.tail(50))
    st.download_button(label="전처리된 공개 데이터 CSV 다운로드", data=public_csv_bytes, file_name="gistemp_preprocessed.csv", mime="text/csv")

# -------------------------
# 사용자 입력 대시보드
# -------------------------
st.header("2. 사용자 입력 대시보드 (프롬프트 Input 섹션 기반)")

# 이 프롬프트의 Input 섹션이 비어있는 것으로 가정. 앱은 실행 중 파일 업로드나 텍스트 입력을 요구하지 않음.
# 따라서 'Input'이 제공되지 않았을 때 사용할 예시 사용자 데이터를 내부 포함.
# (만약 사용자가 이후 Input을 제공하면 본 코드를 수정하여 해당 CSV 텍스트를 여기에 직접 붙여넣도록 함)

# 예시 사용자 데이터 (date, value, group)
# - 사용자는 본 예시 대신 자신의 CSV/이미지/설명을 Input 섹션에 제공할 수 있음.
example_user_csv = """date,value,group
2023-01-01,120,A
2023-02-01,130,A
2023-03-01,125,A
2023-01-01,50,B
2023-02-01,55,B
2023-03-01,60,B
2023-04-01,70,B
2024-05-01,80,A
2025-10-01,999,A
"""

# NOTE: 위 데이터에는 미래(예: 2025-10-01) 샘플이 있어 전처리에서 제거됨(로컬 현재일 기준)
user_df = pd.read_csv(StringIO(example_user_csv))
# 전처리 사용자 데이터 (표준화)
def preprocess_user_df(df):
    df = df.copy()
    # 표준 컬럼 확인/대응
    if 'date' not in df.columns:
        # 시도: 첫 컬럼을 date로 가정
        df = df.rename(columns={df.columns[0]:'date'})
    if 'value' not in df.columns:
        # 시도: 두번째 컬럼을 value로 가정
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[1]:'value'})
    # parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    # remove future dates
    df = df[df['date'].dt.date <= TODAY]
    # numeric value
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    # fill group if missing
    if 'group' not in df.columns:
        df['group'] = '기본'
    df = df.drop_duplicates(subset=['date','group'])
    # missing interpolation per group (time-based)
    df = df.sort_values(['group','date']).reset_index(drop=True)
    df['value'] = df.groupby('group')['value'].apply(lambda s: s.interpolate(method='time', limit_direction='both'))
    return df

user_df = preprocess_user_df(user_df)

# 사이드바: 사용자 데이터 관련 자동 구성
with st.sidebar.expander("사용자 데이터 설정", expanded=False):
    st.write("사용자 입력 데이터는 이 프롬프트의 Input 섹션에서 제공한 CSV/이미지/설명만 사용합니다.")
    smoothing_user = st.checkbox("사용자 데이터 이동평균(기간 선택)", value=True)
    user_window = st.slider("이동평균 기간(개월)", 1, 12, 3)

if user_df.empty:
    st.warning("사용자 입력 데이터가 존재하지 않습니다. (Input 섹션에 데이터를 추가하면 해당 데이터로 대시보드가 구성됩니다.)")
    st.info("현재는 내장 예시 데이터를 사용 중입니다.")
else:
    st.subheader("사용자 데이터 시각화 (예시)")
    # 집단(group) 수에 따라 자동으로 차트 선택
    groups = user_df['group'].unique()
    if len(groups) == 1:
        # 단일 시계열 -> 꺾은선/면적
        df_u = user_df.copy().sort_values('date')
        if smoothing_user and user_window > 1:
            df_u['smoothed'] = df_u['value'].rolling(window=user_window, min_periods=1, center=True).mean()
            y_col = 'smoothed'
        else:
            y_col = 'value'
        fig_u = px.line(df_u, x='date', y=y_col, markers=True, labels={'date':'날짜','value':'값'}, title="사용자 입력 시계열(단일 그룹)")
        st.plotly_chart(fig_u, use_container_width=True)
    else:
        # 다중 그룹 -> 그룹별 꺾은선(범례) 또는 면적 누적
        df_u = user_df.copy().sort_values('date')
        if smoothing_user and user_window > 1:
            df_u['smoothed'] = df_u.groupby('group')['value'].transform(lambda s: s.rolling(window=user_window, min_periods=1, center=True).mean())
            y_col = 'smoothed'
        else:
            y_col = 'value'
        fig_u = px.line(df_u, x='date', y=y_col, color='group', markers=True, labels={'date':'날짜','value':'값','group':'그룹'}, title="사용자 입력 시계열 (그룹별)")
        st.plotly_chart(fig_u, use_container_width=True)

    # 비율형 (group 합계 -> 도넛)
    st.subheader("그룹별 합계 비율")
    group_sum = user_df.groupby('group', as_index=False)['value'].sum()
    fig_pie = px.pie(group_sum, values='value', names='group', hole=0.45, title="그룹별 값 비율 (도넛)")
    st.plotly_chart(fig_pie, use_container_width=True)

    # 지도 시각화: 만약 'lat' & 'lon' 열이 있으면 지도 자동 구성
    if {'lat','lon'}.issubset(user_df.columns):
        st.subheader("위치 기반 시각화 (지도)")
        map_df = user_df.dropna(subset=['lat','lon'])
        st.map(map_df.rename(columns={'lat':'latitude','lon':'longitude'})[['latitude','longitude']])
    else:
        st.info("사용자 데이터에 'lat' 및 'lon' 열이 없으므로 지도 시각화는 생략합니다.")

    # 사용자 데이터 표 + 다운로드
    st.subheader("전처리된 사용자 데이터 표 / CSV 다운로드")
    st.dataframe(user_df)
    st.download_button("사용자 데이터 CSV 다운로드", data=user_df.to_csv(index=False).encode('utf-8'), file_name="user_data_preprocessed.csv", mime="text/csv")

# -------------------------
# 추가 도구 & 도움말 섹션
# -------------------------
st.markdown("---")
st.header("도움말 및 추가 안내 (간단 요약)")

st.markdown("""
- 공개 데이터는 NASA GISS GISTEMP의 공식 CSV를 시도하여 불러옵니다.
  - CSV URL: `https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv`
  - (참고) 다운로드 실패 시 예시 데이터로 자동 대체하고 화면에 안내합니다.
- 사용자 데이터: 이 프롬프트의 Input 섹션에서 제공된 파일/텍스트만 사용합니다. 현재 Input이 비어있어 내장 예시 데이터로 동작 시연합니다.
- kaggle API 사용 안내 (참고)
  1. Kaggle 계정 생성 후 API 토큰(kaggle.json) 다운로드
  2. Codespaces / 로컬 환경에 `~/.kaggle/kaggle.json`으로 위치시킵니다.
  3. `pip install kaggle` 후 `kaggle datasets download -d <dataset>` 사용.
  4. 보안상 토큰은 공개 저장소에 올리지 마세요.
""")

st.caption("앱 버전: Streamlit + GitHub Codespaces 데모 — 모든 라벨/버튼은 한국어로 작성되었습니다.")
