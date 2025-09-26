# --- 라이브러리 임포트 ---
import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

# --- 0. 유틸리티 및 통계 계산 함수 ---

def format_p_value(p):
    """p-값을 논문 형식에 맞게 별표(*)로 변환하는 함수"""
    if p < 0.001:
        return f"{p:.3f} (***)"
    elif p < 0.01:
        return f"{p:.3f} (**)"
    elif p < 0.05:
        return f"{p:.3f} (*)"
    else:
        return f"{p:.3f}"

def calculate_paired_stats(pre_data, post_data):
    """대응표본 t-검정, 효과크기(Cohen's dz), 95% 신뢰구간을 계산하는 함수"""
    temp_df = pd.DataFrame({'pre': pre_data, 'post': post_data}).dropna()
    pre = temp_df['pre']
    post = temp_df['post']
    
    if len(pre) < 2:  # 데이터가 부족하면 계산 불가
        return {}

    # 통계량 계산
    t_stat, p_value = stats.ttest_rel(pre, post)
    diff = post - pre
    n = len(diff)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    
    # Cohen's dz 계산
    cohen_dz = mean_diff / sd_diff if sd_diff != 0 else 0
    
    # 95% 신뢰구간 계산
    se_diff = sd_diff / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_diff - t_critical * se_diff
    ci_high = mean_diff + t_critical * se_diff
    
    return {
        '사전 평균': np.mean(pre),
        '사후 평균': np.mean(post),
        '사전 표준편차': np.std(pre, ddof=1),
        '사후 표준편차': np.std(post, ddof=1),
        '평균 차이': mean_diff,
        't-값': t_stat,
        'p-값': p_value,
        "Cohen's dz": cohen_dz,
        '95% CI 하한': ci_low,
        '95% CI 상한': ci_high,
    }

# --- 1. 앱 기본 설정 ---
st.set_page_config(layout="wide", page_title="대응표본 t-검정 분석기")

# --- 2. 사이드바: 파일 업로드 및 분석 옵션 설정 ---
with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요.", type=['csv'])
    
    df_original = None
    if uploaded_file:
        try:
            df_original = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            st.success("파일 업로드 완료!")
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    if df_original is not None:
        st.header("2. 분석 옵션 설정")
        analysis_unit = st.radio(
            "분석 단위 선택",
            ('문항 단위', '요인 단위'),
            help="'문항 단위'는 개별 변수를 그대로 사용합니다. '요인 단위'는 여러 문항을 합산/평균하여 새로운 요인 변수를 만들어 분석합니다."
        )

        if analysis_unit == '요인 단위':
            agg_method = st.radio(
                "요인 점수 집계 방식",
                ('평균(mean)', '합산(sum)'),
                horizontal=True,
                help="요인을 구성하는 하위 문항들의 점수를 평균낼지 합산할지 선택합니다."
            )

            use_reverse = st.checkbox("역문항 처리 사용")
            reverse_items_str = ""
            if use_reverse:
                max_score = st.number_input("리커트 척도 최대 점수", min_value=1, value=5, step=1)
                reverse_items_str = st.text_area(
                    "역문항 목록 (쉼표로 구분)",
                    placeholder="예: 사전3, 사전5, 사후3, 사후5"
                )

            st.subheader("요인-문항 매핑 입력")
            factor_map_str = st.text_area(
                "분석할 요인 정보를 JSON 형식으로 입력하세요.",
                height=250,
                placeholder='''{
    "주도성": {
        "pre": ["사전1", "사전2", "사전3"],
        "post": ["사후1", "사후2", "사후3"]
    },
    "AI리터러시": {
        "pre": ["사전4", "사전5"],
        "post": ["사후4", "사후5"]
    }
}'''
            )
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 12px; color: grey;">
            © 2025 AI. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )


# --- 3. 메인 화면 ---
st.title("📄 교육연구대회용 사전-사후 데이터 분석")
st.subheader("대응표본 t-검정, 효과크기, 시각화 (문항/요인 단위 분석 지원)")

if df_original is not None:
    tab1, tab2 = st.tabs(["분석 결과", "시각화"])

    with tab1:
        st.header("📊 분석 결과 요약")
        df = df_original.copy()
        
        # --- 문항 단위 분석 UI ---
        if 'analysis_unit' in locals() and analysis_unit == '문항 단위':
            st.subheader("🔍 분석할 변수 선택 (문항 단위)")
            col1, col2 = st.columns(2)
            with col1:
                pre_vars = st.multiselect('사전 검사 변수 선택', df.columns.tolist(), key="pre_item")
            with col2:
                post_vars = st.multiselect('사후 검사 변수 선택', df.columns.tolist(), key="post_item")

            if pre_vars and post_vars and len(pre_vars) != len(post_vars):
                st.warning("⚠️ 사전-사후 검사 변수의 개수를 동일하게 맞춰주세요.")

            if st.button('🚀 문항 단위 분석 실행', type="primary", key="run_item_analysis"):
                results = []
                if not pre_vars or not post_vars or len(pre_vars) != len(post_vars):
                    st.error("분석을 실행하려면 사전-사후 변수 쌍을 올바르게 선택해주세요.")
                else:
                    for pre_var, post_var in zip(pre_vars, post_vars):
                        stats_result = calculate_paired_stats(df[pre_var], df[post_var])
                        if stats_result:
                            stats_result['사전 변수'] = pre_var
                            stats_result['사후 변수'] = post_var
                            results.append(stats_result)
                            
                    if results:
                        st.session_state['results'] = results
                        st.session_state['df_for_plotting'] = df
                        st.session_state['analysis_unit'] = '문항 단위'
                        st.session_state['plot_vars'] = list(zip(pre_vars, post_vars))
                        st.experimental_rerun()
                    else:
                        st.warning("분석 결과가 없습니다. 데이터를 확인해주세요.")

        # --- 요인 단위 분석 UI 및 실행 로직 ---
        elif 'analysis_unit' in locals() and analysis_unit == '요인 단위':
            if st.button('🚀 요인 단위 분석 실행', type="primary", key="run_factor_analysis"):
                results = []
                df_for_plotting = pd.DataFrame()
                
                try:
                    if not factor_map_str:
                        st.error("요인-문항 매핑 정보를 입력해주세요.")
                        st.stop()
                    factor_map = json.loads(factor_map_str)

                    if 'use_reverse' in locals() and use_reverse and reverse_items_str:
                        reverse_items = [item.strip() for item in reverse_items_str.split(',')]
                        for item in reverse_items:
                            if item in df.columns:
                                df[item] = (max_score + 1) - df[item]
                            else:
                                st.warning(f"'{item}' 컬럼이 데이터에 없어 역코딩을 건너뜁니다.")
                                
                    analysis_pairs = []
                    for factor, items in factor_map.items():
                        pre_cols, post_cols = items.get('pre', []), items.get('post', [])
                        
                        if not pre_cols or not post_cols:
                            st.error(f"'{factor}' 요인에 'pre' 또는 'post' 문항 목록이 없습니다.")
                            continue
                        
                        factor_df_temp = df[pre_cols + post_cols].dropna()

                        if agg_method == '평균(mean)':
                            pre_score = factor_df_temp[pre_cols].mean(axis=1)
                            post_score = factor_df_temp[post_cols].mean(axis=1)
                        else: # '합산(sum)'
                            pre_score = factor_df_temp[pre_cols].sum(axis=1)
                            post_score = factor_df_temp[post_cols].sum(axis=1)

                        stats_result = calculate_paired_stats(pre_score, post_score)
                        if stats_result:
                            stats_result['요인'] = factor
                            results.append(stats_result)
                            
                        plot_temp_df = pd.DataFrame({
                            f"{factor}_사전": pre_score,
                            f"{factor}_사후": post_score
                        })
                        df_for_plotting = pd.concat([df_for_plotting, plot_temp_df], axis=1)
                        analysis_pairs.append((f"{factor}_사전", f"{factor}_사후"))

                    if results:
                        st.session_state['results'] = results
                        st.session_state['df_for_plotting'] = df_for_plotting
                        st.session_state['analysis_unit'] = '요인 단위'
                        st.session_state['plot_vars'] = analysis_pairs
                        st.experimental_rerun()
                    else:
                        st.warning("분석 결과가 없습니다. 데이터를 확인해주세요.")
                
                except json.JSONDecodeError:
                    st.error("JSON 형식이 올바르지 않습니다. 예시를 참고하여 수정해주세요.")
                except Exception as e:
                    st.error(f"요인 분석 중 오류가 발생했습니다: {e}")

        # --- 결과 표시 로직 (세션 상태 활용) ---
        if 'results' in st.session_state:
            results_df = pd.DataFrame(st.session_state['results'])
            
            if st.session_state['analysis_unit'] == '요인 단위':
                cols_order = ['요인', '사전 평균', '사후 평균', '평균 차이', '사전 표준편차', '사후 표준편차',
                              't-값', 'p-값', "Cohen's dz", '95% CI 하한', '95% CI 상한']
            else: # 문항 단위
                cols_order = ['사전 변수', '사후 변수', '사전 평균', '사후 평균', '평균 차이', '사전 표준편차', '사후 표준편차',
                              't-값', 'p-값', "Cohen's dz", '95% CI 하한', '95% CI 상한']
            
            results_df['유의성'] = results_df['p-값'].apply(format_p_value)
            results_df = results_df[[col for col in cols_order if col in results_df.columns] + ['유의성']]
            
            format_dict = {col: '{:.3f}' for col in results_df.columns if results_df[col].dtype == 'float64'}
            st.dataframe(results_df.style.format(format_dict), use_container_width=True)
            
            st.info("p-값 형식: p < .05 (*), p < .01 (**), p < .001 (***)")
            
            csv_data = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 결과 테이블 CSV로 다운로드",
                data=csv_data,
                file_name="t-test_analysis_results.csv",
                mime="text/csv",
            )
            
            # --- 주요 통계량 시각화 ---
            st.subheader("💡 주요 통계량")
            if not results_df.empty:
                col_pre, col_post, col_diff, col_p = st.columns(4)
                with col_pre:
                    st.metric("사전 평균", f"{results_df['사전 평균'].mean():.2f}")
                with col_post:
                    st.metric("사후 평균", f"{results_df['사후 평균'].mean():.2f}")
                with col_diff:
                    st.metric("평균 차이", f"{results_df['평균 차이'].mean():.2f}")
                with col_p:
                    st.metric("평균 p-값", format_p_value(results_df['p-값'].mean()))

    with tab2:
        st.header("📈 시각화 자료")
        if 'df_for_plotting' in st.session_state and 'plot_vars' in st.session_state:
            df_for_plotting = st.session_state['df_for_plotting']
            plot_vars = st.session_state['plot_vars']
            
            for pre_var, post_var in plot_vars:
                # 탭 형태로 개별 분석 결과 표시
                var_name = pre_var.replace('_사전', '')
                with st.expander(f"**{var_name}** 분석 시각화"):
                    # 1. Raincloud Plot
                    plot_df_long = pd.DataFrame({
                        '점수': pd.concat([df_for_plotting[pre_var], df_for_plotting[post_var]], ignore_index=True),
                        '시점': [f'사전'] * len(df_for_plotting) + [f'사후'] * len(df_for_plotting)
                    })
                    fig_rain = px.violin(
                        plot_df_long, y='점수', x='시점', color='시점',
                        box=True, points='all', color_discrete_map={'사전': 'blue', '사후': 'orange'}
                    )
                    fig_rain.update_layout(
                        title_text=f"<b>{var_name} 사전-사후 분포 비교 (Raincloud Plot)</b>",
                        xaxis_title='검사 시점', yaxis_title='점수', showlegend=False
                    )
                    st.plotly_chart(fig_rain, use_container_width=True)

                    # 2. Paired Line Plot
                    fig_line = go.Figure()
                    
                    colors = px.colors.qualitative.Plotly
                    
                    for i in range(len(df_for_plotting)):
                        color = colors[i % len(colors)]
                        
                        fig_line.add_trace(go.Scatter(
                            x=['사전', '사후'],
                            y=[df_for_plotting.loc[i, pre_var], df_for_plotting.loc[i, post_var]],
                            mode='lines+markers',
                            line=dict(color=color, width=1.5),
                            marker=dict(color=color),
                            showlegend=False
                        ))
                    fig_line.update_layout(
                        title_text=f"<b>{var_name} 개별 데이터 변화 추이</b>",
                        xaxis_title='검사 시점',
                        yaxis_title='점수'
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                    st.markdown("---")
        else:
            st.info("분석 결과를 보려면 '분석 결과' 탭에서 분석을 먼저 실행해주세요.")
            
else:
    st.info("👈 사이드바에서 CSV 파일을 업로드하여 분석을 시작하세요.")
