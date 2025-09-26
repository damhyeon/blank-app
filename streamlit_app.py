# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, date

# --- 폰트 설정 (없으면 자동 생략) ---
try:
    # Streamlit 기본 폰트 설정 (전역 적용은 불가, 개별 요소에 적용 시도)
    st.markdown(
        """
        <style>
        @font-face {
            font-family: 'Pretendard-Bold';
            src: url('fonts/Pretendard-Bold.ttf') format('truetype');
        }
        body, .stApp, h1, h2, h3, h4, h5, h6, label, p, .css-1lcbmhc, .css-1qxt0et, .css-fg4pbf, .css-1dp5vir {
            font-family: 'Pretendard-Bold', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Matplotlib 및 Seaborn 폰트 설정
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fe = fm.FontEntry(fname='fonts/Pretendard-Bold.ttf', name='Pretendard-Bold')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.family'] = 'Pretendard-Bold'
    plt.rcParams['axes.unicode_minus'] = False # 음수 부호 깨짐 방지
except Exception:
    pass # 폰트 파일이 없으면 에러 발생하므로 무시

# --- 캐싱 데코레이터 ---
@st.cache_data
def get_noaa_sea_level_data():
    """
    NOAA 해수면 상승 데이터를 가져옵니다.
    출처: https://www.ncei.noaa.gov/products/sea-level-rise
    """
    try:
        # Simplified data fetching, actual NOAA API might require more complex queries
        # For demonstration, using a placeholder CSV or a direct link if available
        url = "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_table.csv"
        df = pd.read_csv(url, skiprows=2, header=0) # Adjust skiprows/header based on actual CSV
        df.columns = [col.strip() for col in df.columns] # Remove whitespace from column names
        df = df.rename(columns={
            'Year': 'year',
            'Month': 'month',
            'Day': 'day',
            'GMSL (mm)': 'value',
            'GMSL Uncertainty (mm)': 'uncertainty'
        })
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df = df[['date', 'value', 'uncertainty']]
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        df = df[df['date'] <= pd.to_datetime(date.today())] # 오늘 이후 데이터 제거
        st.success("NOAA 해수면 데이터를 성공적으로 로드했습니다.")
        return df
    except Exception as e:
        st.warning(f"NOAA 해수면 데이터 로드 실패: {e}. 예시 데이터로 대체합니다.")
        # 예시 데이터 (1993년부터 2024년까지 월별 데이터)
        start_date = datetime(1993, 1, 1)
        end_date = datetime(2024, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        values = np.linspace(0, 100, len(dates)) + np.random.randn(len(dates)) * 5
        example_df = pd.DataFrame({'date': dates, 'value': values.round(2)})
        example_df['uncertainty'] = np.random.uniform(1, 3, len(dates)).round(2)
        example_df = example_df[example_df['date'] <= pd.to_datetime(date.today())]
        return example_df

@st.cache_data
def get_antarctic_ice_melt_data():
    """
    남극 빙하 데이터 예시 (더미 데이터 사용)
    실제 데이터 출처: NASA, NSIDC 등
    """
    try:
        # Placeholder for actual data fetching from NASA or NSIDC
        # For demonstration, create synthetic data
        start_date = datetime(1979, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='YS') # Yearly data
        ice_mass = 15000 - np.cumsum(np.random.rand(len(dates)) * 50) # Decreasing trend
        surface_temp = 273.15 + np.cumsum(np.random.rand(len(dates)) * 0.05) - 273.15 + np.random.randn(len(dates)) * 0.5 # Increasing trend
        
        df = pd.DataFrame({
            'date': dates,
            'ice_mass_Gt': ice_mass.round(2),
            'surface_temperature_C': surface_temp.round(2)
        })
        df = df[df['date'] <= pd.to_datetime(date.today())]
        st.success("남극 빙하 및 기온 데이터를 성공적으로 로드했습니다 (예시 데이터).")
        return df
    except Exception as e:
        st.warning(f"남극 빙하 및 기온 데이터 로드 실패: {e}. 예시 데이터로 대체합니다.")
        start_date = datetime(1979, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='YS') # Yearly data
        ice_mass = 15000 - np.cumsum(np.random.rand(len(dates)) * 50)
        surface_temp = -20 + np.cumsum(np.random.rand(len(dates)) * 0.1) + np.random.randn(len(dates)) * 0.5
        df = pd.DataFrame({
            'date': dates,
            'ice_mass_Gt': ice_mass.round(2),
            'surface_temperature_C': surface_temp.round(2)
        })
        df = df[df['date'] <= pd.to_datetime(date.today())]
        return df

@st.cache_data
def get_ghg_emissions_data():
    """
    온실가스 배출량 데이터 예시 (더미 데이터 사용)
    실제 데이터 출처: World Bank, NOAA ESRL 등
    """
    try:
        # Placeholder for actual data fetching (e.g., from World Bank API)
        # For demonstration, create synthetic data
        start_year = 1990
        end_year = 2022
        years = pd.to_datetime([f"{y}-01-01" for y in range(start_year, end_year + 1)])
        co2_emissions = np.linspace(300, 420, len(years)) + np.random.randn(len(years)) * 5
        ch4_emissions = np.linspace(1700, 1900, len(years)) + np.random.randn(len(years)) * 20
        n2o_emissions = np.linspace(300, 330, len(years)) + np.random.randn(len(years)) * 5

        df = pd.DataFrame({
            'date': years,
            'CO2_ppm': co2_emissions.round(2),
            'CH4_ppb': ch4_emissions.round(2),
            'N2O_ppb': n2o_emissions.round(2)
        })
        df = df[df['date'] <= pd.to_datetime(date.today())]
        st.success("온실가스 배출량 데이터를 성공적으로 로드했습니다 (예시 데이터).")
        return df
    except Exception as e:
        st.warning(f"온실가스 배출량 데이터 로드 실패: {e}. 예시 데이터로 대체합니다.")
        start_year = 1990
        end_year = 2022
        years = pd.to_datetime([f"{y}-01-01" for y in range(start_year, end_year + 1)])
        co2_emissions = np.linspace(300, 420, len(years)) + np.random.randn(len(years)) * 5
        ch4_emissions = np.linspace(1700, 1900, len(years)) + np.random.randn(len(years)) * 20
        n2o_emissions = np.linspace(300, 330, len(years)) + np.random.randn(len(years)) * 5

        df = pd.DataFrame({
            'date': years,
            'CO2_ppm': co2_emissions.round(2),
            'CH4_ppb': ch4_emissions.round(2),
            'N2O_ppb': n2o_emissions.round(2)
        })
        df = df[df['date'] <= pd.to_datetime(date.today())]
        return df

# --- UI 설정 ---
st.set_page_config(layout="wide", page_title="전 지구 해수면 상승 분석 대시보드")

st.title("🌊 전 지구 해수면 상승 및 기후 변화 대시보드")

# --- 사이드바 ---
st.sidebar.header("⚙️ 대시보드 설정")

with st.sidebar.expander("해수면 데이터 옵션", expanded=True):
    sea_level_start_year, sea_level_end_year = st.slider(
        '기간 선택',
        1993, datetime.now().year, (1993, datetime.now().year),
        key='sea_level_period'
    )
    show_annotations = st.checkbox('주석 표시', value=True)

with st.sidebar.expander("분석 옵션", expanded=True):
    analysis_start_year, analysis_end_year = st.slider(
        '분석 기간',
        1998, datetime.now().year, (1998, datetime.now().year),
        key='analysis_period'
    )
    show_correlation = st.checkbox('상관관계 분석 표시')
    moving_avg_window = st.slider('이동평균 변동수', 1, 16, 4, key='moving_avg_window')

with st.sidebar.expander("온실가스 분석 옵션", expanded=True):
    ghg_type = st.selectbox('온실가스 종류 선택', ['CO2', 'CH4', 'N2O'], key='ghg_type')
    ghg_start_year, ghg_end_year = st.slider(
        '온실가스 분석 기간',
        1998, 2022, (1998, 2022),
        key='ghg_analysis_period'
    )

# --- 메인 콘텐츠 탭 ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 전 지구 해수면 상승 데이터",
    "🧊 남극 빙하 및 기온 데이터",
    "💨 온실가스 배출량 분석",
    "🌱 글로벌 해수면 복원 방안",
    "📍 평균 해안선 (사용자 입력)",
    "📊 전 지구 해수면 상승 (사용자 입력)"
])

# --- 탭 1: 전 지구 해수면 상승 데이터 (공개 데이터) ---
with tab1:
    st.header("🌍 전 지구 해수면 상승 데이터")
    st.markdown("---")

    sea_level_df = get_noaa_sea_level_data()
    
    if not sea_level_df.empty:
        filtered_sea_level_df = sea_level_df[
            (sea_level_df['date'].dt.year >= sea_level_start_year) &
            (sea_level_df['date'].dt.year <= sea_level_end_year)
        ].copy() # SettingWithCopyWarning 방지
        
        if not filtered_sea_level_df.empty:
            # 이동평균 계산
            filtered_sea_level_df['moving_avg'] = filtered_sea_level_df['value'].rolling(window=moving_avg_window, center=True).mean()

            # 주요 지표 계산
            current_sea_level = filtered_sea_level_df['value'].iloc[-1] if not filtered_sea_level_df.empty else 0
            
            # 연평균 상승률 계산 (선형 회귀)
            if len(filtered_sea_level_df) > 1:
                from scipy.stats import linregress
                x = (filtered_sea_level_df['date'] - filtered_sea_level_df['date'].min()).dt.days / 365.25
                y = filtered_sea_level_df['value']
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                annual_rise_rate = slope # mm/year
            else:
                annual_rise_rate = 0

            total_rise_amount = filtered_sea_level_df['value'].iloc[-1] - filtered_sea_level_df['value'].iloc[0] if len(filtered_sea_level_df) > 1 else 0
            
            st.subheader("💡 주요 지표")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"**현재 해수면 상승량**\n###### {current_sea_level:.2f} mm")
            with col2:
                st.info(f"**연평균 상승률**\n###### {annual_rise_rate:.2f} mm/년")
            with col3:
                st.info(f"**총 상승량**\n###### {total_rise_amount:.2f} mm")
            with col4:
                st.info(f"**측정 기간**\n###### {sea_level_start_year}년 ~ {sea_level_end_year}년")

            st.subheader("📈 전 지구 해수면 상승 추이 (NOAA 데이터)")
            
            fig = px.line(
                filtered_sea_level_df,
                x='date',
                y='value',
                title=f'전 지구 해수면 상승 추이 ({sea_level_start_year}-{sea_level_end_year})',
                labels={'date': '날짜', 'value': '해수면 상승량 (mm)'},
                line_shape='spline',
                height=500,
                color_discrete_sequence=['#1f77b4'] # 파란색
            )
            
            fig.add_trace(go.Scatter(
                x=filtered_sea_level_df['date'],
                y=filtered_sea_level_df['moving_avg'],
                mode='lines',
                name=f'{moving_avg_window}개월 이동평균',
                line=dict(color='#ff7f0e', width=2) # 주황색
            ))

            if show_annotations:
                fig.add_annotation(
                    x=filtered_sea_level_df['date'].iloc[-1],
                    y=filtered_sea_level_df['value'].iloc[-1],
                    text=f"현재: {filtered_sea_level_df['value'].iloc[-1]:.2f} mm",
                    showarrow=True,
                    arrowhead=1,
                    font=dict(size=12, color="red")
                )

            fig.update_layout(
                hovermode="x unified",
                xaxis_title="날짜",
                yaxis_title="해수면 상승량 (mm)",
                font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 주요 지역별 해수면 상승률 (더미 데이터)")
            # 실제 데이터 대신 예시 데이터를 사용하여 지역별 상승률 시각화
            regional_data = {
                '지역': ['태평양', '대서양', '인도양', '북극해', '남극해'],
                '상승률 (mm/년)': [3.5, 3.2, 3.8, 4.1, 2.9]
            }
            regional_df = pd.DataFrame(regional_data)
            
            fig_bar = px.bar(
                regional_df,
                x='지역',
                y='상승률 (mm/년)',
                title='주요 지역별 연평균 해수면 상승률 (가상 데이터)',
                labels={'지역': '주요 해역', '상승률 (mm/년)': '연평균 상승률'},
                color_discrete_sequence=['#d62728'], # 빨간색
                height=400
            )
            fig_bar.update_layout(
                xaxis_title="주요 해역",
                yaxis_title="연평균 상승률 (mm/년)",
                font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("데이터 내보내기")
            st.download_button(
                label="CSV로 다운로드",
                data=filtered_sea_level_df.to_csv(index=False).encode('utf-8'),
                file_name="해수면_상승_데이터.csv",
                mime="text/csv",
            )
        else:
            st.warning(f"{sea_level_start_year}년부터 {sea_level_end_year}년까지의 해수면 데이터가 없습니다.")
    else:
        st.error("NOAA 해수면 데이터를 로드하는 데 실패했습니다. 다시 시도하거나 인터넷 연결을 확인해주세요.")


# --- 탭 2: 남극 빙하 및 기온 데이터 (공개 데이터) ---
with tab2:
    st.header("🧊 남극 빙하 및 기온 데이터")
    st.markdown("---")

    antarctic_df = get_antarctic_ice_melt_data()

    if not antarctic_df.empty:
        filtered_antarctic_df = antarctic_df[
            (antarctic_df['date'].dt.year >= analysis_start_year) &
            (antarctic_df['date'].dt.year <= analysis_end_year)
        ]

        if not filtered_antarctic_df.empty:
            st.subheader("📉 남극 빙하 질량 변화 추이")
            fig_ice = px.line(
                filtered_antarctic_df,
                x='date',
                y='ice_mass_Gt',
                title=f'남극 빙하 질량 변화 ({analysis_start_year}-{analysis_end_year})',
                labels={'date': '날짜', 'ice_mass_Gt': '빙하 질량 (기가톤)'},
                line_shape='spline',
                height=400,
                color_discrete_sequence=['#2ca02c'] # 초록색
            )
            fig_ice.update_layout(
                hovermode="x unified",
                xaxis_title="날짜",
                yaxis_title="빙하 질량 (기가톤)",
                font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
            )
            st.plotly_chart(fig_ice, use_container_width=True)

            st.subheader("🌡️ 남극 표면 기온 변화 추이")
            fig_temp = px.line(
                filtered_antarctic_df,
                x='date',
                y='surface_temperature_C',
                title=f'남극 표면 기온 변화 ({analysis_start_year}-{analysis_end_year})',
                labels={'date': '날짜', 'surface_temperature_C': '표면 기온 (°C)'},
                line_shape='spline',
                height=400,
                color_discrete_sequence=['#d62728'] # 빨간색
            )
            fig_temp.update_layout(
                hovermode="x unified",
                xaxis_title="날짜",
                yaxis_title="표면 기온 (°C)",
                font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

            if show_correlation:
                st.subheader("🤝 빙하 질량 및 기온 상관관계 분석")
                # 빙하 질량과 기온 간의 상관관계 시각화 (산점도)
                fig_corr = px.scatter(
                    filtered_antarctic_df,
                    x='surface_temperature_C',
                    y='ice_mass_Gt',
                    trendline='ols', # OLS (Ordinary Least Squares) 회귀선 추가
                    title='남극 표면 기온 vs. 빙하 질량 (상관관계)',
                    labels={'surface_temperature_C': '표면 기온 (°C)', 'ice_mass_Gt': '빙하 질량 (기가톤)'},
                    height=400,
                    color_discrete_sequence=['#9467bd'] # 보라색
                )
                fig_corr.update_layout(
                    xaxis_title="표면 기온 (°C)",
                    yaxis_title="빙하 질량 (기가톤)",
                    font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                correlation = filtered_antarctic_df['surface_temperature_C'].corr(filtered_antarctic_df['ice_mass_Gt'])
                st.info(f"빙하 질량과 표면 기온 간의 상관계수: **{correlation:.2f}**")
                st.markdown("음의 상관관계는 기온이 상승할수록 빙하 질량이 감소함을 의미합니다.")
            
            st.subheader("데이터 내보내기")
            st.download_button(
                label="CSV로 다운로드",
                data=filtered_antarctic_df.to_csv(index=False).encode('utf-8'),
                file_name="남극_빙하_기온_데이터.csv",
                mime="text/csv",
            )
        else:
            st.warning(f"{analysis_start_year}년부터 {analysis_end_year}년까지의 남극 데이터가 없습니다.")
    else:
        st.error("남극 빙하 및 기온 데이터를 로드하는 데 실패했습니다.")

# --- 탭 3: 온실가스 배출량 분석 (공개 데이터) ---
with tab3:
    st.header("💨 온실가스 배출량 분석")
    st.markdown("---")

    ghg_df = get_ghg_emissions_data()

    if not ghg_df.empty:
        filtered_ghg_df = ghg_df[
            (ghg_df['date'].dt.year >= ghg_start_year) &
            (ghg_df['date'].dt.year <= ghg_end_year)
        ]

        if not filtered_ghg_df.empty:
            st.subheader(f"📈 글로벌 {ghg_type} 배출량 추이")
            y_col = f'{ghg_type}_ppm' if ghg_type == 'CO2' else f'{ghg_type}_ppb'
            y_label = '농도 (ppm)' if ghg_type == 'CO2' else '농도 (ppb)'

            fig_ghg = px.line(
                filtered_ghg_df,
                x='date',
                y=y_col,
                title=f'글로벌 {ghg_type} 농도 변화 ({ghg_start_year}-{ghg_end_year})',
                labels={'date': '날짜', y_col: y_label},
                line_shape='spline',
                height=500,
                color_discrete_sequence=['#ff7f0e'] # 주황색
            )
            fig_ghg.update_layout(
                hovermode="x unified",
                xaxis_title="날짜",
                yaxis_title=y_label,
                font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
            )
            st.plotly_chart(fig_ghg, use_container_width=True)
            
            st.subheader("데이터 내보내기")
            st.download_button(
                label="CSV로 다운로드",
                data=filtered_ghg_df.to_csv(index=False).encode('utf-8'),
                file_name="온실가스_배출량_데이터.csv",
                mime="text/csv",
            )
        else:
            st.warning(f"{ghg_start_year}년부터 {ghg_end_year}년까지의 {ghg_type} 데이터가 없습니다.")
    else:
        st.error("온실가스 배출량 데이터를 로드하는 데 실패했습니다.")

# --- 탭 4: 글로벌 해수면 복원 방안 (더미 콘텐츠) ---
with tab4:
    st.header("🌱 글로벌 해수면 복원 방안")
    st.markdown("---")
    st.write("""
    해수면 상승은 심각한 기후 변화의 결과이며, 이를 완화하고 적응하기 위한 다양한 방안이 논의되고 있습니다.
    """)
    st.subheader("1. 온실가스 배출량 감축")
    st.markdown("""
    *   **재생에너지 전환**: 화석 연료 사용을 줄이고 태양광, 풍력 등 재생에너지 비중 확대.
    *   **에너지 효율 향상**: 산업, 건물, 운송 등 모든 부문에서 에너지 소비 효율을 높이는 기술 개발 및 적용.
    *   **탄소 포집 및 저장(CCS)**: 대기 중 탄소를 직접 포집하여 저장하거나 재활용하는 기술 상용화.
    """)
    st.subheader("2. 해안선 보호 및 적응 전략")
    st.markdown("""
    *   **자연 기반 해안선 보호**: 맹그로브 숲 조성, 산호초 복원 등 자연 생태계를 활용한 해안선 보호.
    *   **방조제 및 제방 건설**: 해수면 상승에 취약한 지역에 인공 구조물 건설.
    *   **도시 계획 재조정**: 해수면 상승 위험 지역의 개발을 제한하고, 거주지 및 인프라를 높은 지대로 이전.
    """)
    st.subheader("3. 국제 협력 및 정책 강화")
    st.markdown("""
    *   **파리 협정 이행**: 국제 사회가 온실가스 감축 목표를 달성하기 위한 노력 강화.
    *   **기후 변화 기금 마련**: 개발도상국의 기후 변화 적응 및 완화 노력 지원.
    *   **기술 공유 및 연구 투자**: 해수면 상승 예측 모델 고도화 및 친환경 기술 개발을 위한 국제적인 협력 증진.
    """)
    st.info("이 섹션의 내용은 예시이며, 실제 복원 방안은 더욱 복잡하고 다층적입니다.")
    
    # 이미지 추가 (더미)
    st.subheader("관련 이미지")
    # 이미지 생성 프롬프트
    st.markdown("해수면 상승에 대응하는 다양한 방안들을 시각적으로 보여주는 이미지를 생성합니다. 재생에너지, 해안선 보호 구조물, 그리고 국제 협력하는 모습을 한 이미지에 담아주세요.")
    # Image placeholder for proactive illustration
    st.image("https://via.placeholder.com/800x400.png?text=Global+Sea+Level+Solutions", caption="해수면 상승 대응 방안 (예시 이미지)")


# --- 사용자 입력 데이터 처리 (탭 5, 6) ---
# 평균 해안선 데이터 (이미지 설명 기반 더미 데이터)
# 이미지 설명: "평균 해안선" 탭을 만들어줘. -> 지도 시각화 필요.
# 이 탭에서는 이미지 생성을 직접 수행하지 않고, 이미지 설명을 기반으로 데이터와 시각화를 만듭니다.
# 사용자가 직접 데이터를 업로드하는 것이 아니므로, 하드코딩된 예시 데이터를 사용합니다.
with tab5:
    st.header("📍 평균 해안선 (사용자 입력 데이터 기반)")
    st.markdown("---")
    st.write("사용자 입력(프롬프트 설명)을 기반으로 생성된 평균 해안선 변화 대시보드입니다.")
    st.info("이 데이터는 사용자의 '평균 해안선' 탭 생성 요구사항을 바탕으로 생성된 예시 데이터입니다.")

    # 더미 데이터 생성 (전세계 주요 도시 및 가상의 해수면 변화 영향)
    coastline_data = {
        '도시': ['서울', '부산', '뉴욕', '도쿄', '상하이', '런던', '베니스', '싱가포르', '시드니', '마이애미'],
        '위도': [37.5665, 35.1796, 40.7128, 35.6762, 31.2304, 51.5074, 45.4408, 1.3521, -33.8688, 25.7617],
        '경도': [126.9780, 129.0756, -74.0060, 139.6503, 121.4737, -0.1278, 12.3155, 103.8198, 151.2093, -80.1918],
        '예상_해수면_변화_mm_2050년': [30, 70, 60, 50, 80, 20, 100, 90, 40, 120],
        '취약도': ['낮음', '중간', '중간', '낮음', '높음', '낮음', '매우 높음', '높음', '낮음', '매우 높음']
    }
    coastline_df = pd.DataFrame(coastline_data)

    st.subheader("주요 해안 도시별 예상 해수면 변화 (2050년)")
    
    # 지도 시각화
    fig_map = px.scatter_mapbox(
        coastline_df,
        lat="위도",
        lon="경도",
        color="예상_해수면_변화_mm_2050년",
        size="예상_해수면_변화_mm_2050년",
        hover_name="도시",
        hover_data={"취약도": True, "예상_해수면_변화_mm_2050년": ":.0f mm"},
        color_continuous_scale=px.colors.sequential.Plasma,
        zoom=1,
        height=600,
        title="주요 해안 도시별 예상 해수면 변화 (2050년)"
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig_map.update_layout(
        font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("취약도별 도시 분포")
    fig_pie = px.pie(
        coastline_df,
        names='취약도',
        title='해안 도시 취약도 분포 (가상 데이터)',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=400
    )
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(
        font_family="Pretendard-Bold" if "Pretendard-Bold" in plt.rcParams['font.family'] else "sans-serif"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("데이터 내보내기")
    st.download_button(
        label="CSV로 다운로드",
        data=coastline_df.to_csv(index=False).encode('utf-8'),
        file_name="평균_해안선_예측_데이터.csv",
        mime="text/csv",
    )


# 전 지구 해수면 상승 추이 (사용자 입력 데이터)
# CSV 데이터 입력:
# "날짜,해수면_변화_mm
# 1993-01-01,0.0
# 1993-02-01,0.2
# ...
) # 