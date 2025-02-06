import os
import numpy as np
import pandas as pd  # ✅ 데이터 저장을 위한 pandas 추가
import streamlit as st
from keras.models import load_model  
from PIL import Image, ImageOps
import plotly.express as px  
import time

# ✅ 데이터 저장용 CSV 파일 경로
data_file = "data.csv"

# ✅ 기존 데이터 불러오기 (없으면 새 파일 생성)
if os.path.exists(data_file):
    df = pd.read_csv(data_file, encoding='utf-8-sig')
else:
    df = pd.DataFrame(columns=["날짜", "예측된 관상", "확률"])
    df.to_csv(data_file, index=False, encoding='utf-8-sig')  # ✅ 빈 파일이라도 먼저 생성

def save_prediction(prediction_class, confidence_score):
    """ 예측된 데이터를 CSV 파일에 저장하는 함수 """
    global df  # ✅ df를 전역 변수로 사용하여 유지
    new_data = pd.DataFrame([[time.strftime("%Y-%m-%d %H:%M:%S"), prediction_class, confidence_score]],
                            columns=["날짜", "예측된 관상", "확률"])
    
    df = pd.concat([df, new_data], ignore_index=True)  # ✅ 기존 데이터에 새로운 데이터 추가
    df.to_csv(data_file, index=False, encoding='utf-8-sig')  # ✅ CSV 파일 업데이트
    st.success("✅ 예측 결과가 성공적으로 저장되었습니다!")

def run_ml():
    st.title("📊 어떤 관상이 많을까?")
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, encoding='utf-8-sig')  # ✅ 최신 데이터 불러오기
    else:
        df = pd.DataFrame(columns=["날짜", "예측된 관상", "확률"])

    if df.empty:
        st.warning("📢 아직 저장된 데이터가 없습니다. 먼저 관상 분석을 진행하세요!")
    else:
        # ✅ 누적 데이터 표시
        st.subheader("📋 분석 데이터")
        st.sort_indataframe(df.head(10))

        # ✅ 관상별 개수 시각화
        count_data = df["예측된 관상"].value_counts().reset_index()
        count_data.columns = ["관상", "개수"]
        fig_bar = px.bar(count_data, x="관상", y="개수", text="개수", title="각 관상의 분석된 개수", color="관상")
        st.subheader("📊 관상별 개수")
        st.plotly_chart(fig_bar, use_container_width=True)

        # ✅ 평균 확률 시각화
        avg_confidence = df.groupby("예측된 관상")["확률"].mean().reset_index()
        fig_pie = px.pie(avg_confidence, names="예측된 관상", values="확률", title="관상별 평균 확률 (%)", hole=0.3)
        st.subheader("📊 관상별 평균 확률")
        st.plotly_chart(fig_pie, use_container_width=True)
