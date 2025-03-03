import os
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model  
from PIL import Image, ImageOps
import plotly.express as px  
import time

from eda import run_eda
from home import run_home
from ml import run_ml  

# 👉 페이지 설정
st.set_page_config(page_title="관상 분석 AI", page_icon="🔮", layout="wide")

# 👉 세련된 스타일 적용
st.markdown("""
    <style>
        /* 전체 배경 색상 */
        body {
            background-color: #f0f2f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* 메인 타이틀 스타일 */
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #FF6347;
            text-align: center;
            padding: 20px 0;
            margin: 0;
        }

        /* 사이드바 스타일 */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
            color: #333;
        }

        /* 사이드바 내 요소 스타일 */
        .sidebar .sidebar-content {
            padding: 10px;
            border-radius: 10px;
        }

        /* 라디오 버튼 스타일 */
        [data-testid="stSidebar"] .stRadio > label {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }

        /* 선택된 라디오 버튼 스타일 */
        [data-testid="stSidebar"] .stRadio > div > label > div[role="radiogroup"] > div {
            border: 2px solid #FF6347;
            background-color: #FF6347;
            color: white;
            border-radius: 10px;
            margin: 5px 0;
            transition: 0.3s;
        }

        /* 호버 효과 */
        [data-testid="stSidebar"] .stRadio > div > label > div[role="radiogroup"] > div:hover {
            background-color: #FFA07A;
            border-color: #FF4500;
            color: white;
        }

        /* 버튼 스타일 */
        .stButton > button {
            background-color: #FF6347;
            color: white;
            font-size: 1.1em;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            box-shadow: 2px 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        /* 버튼 호버 스타일 */
        .stButton > button:hover {
            background-color: #FF4500;
            box-shadow: 4px 6px 12px rgba(0,0,0,0.2);
            transform: translateY(-3px);
        }

      
        
    </style>
""", unsafe_allow_html=True)



# 👉 사이드바 메뉴
menu = st.sidebar.radio("메뉴 선택", ["🏠 홈", "🔍 나는 무슨 관상일까?", "📊 어떤 관상이 많을까?"], label_visibility="collapsed")

# 👉 선택된 메뉴에 따른 화면 전환
if menu == "🏠 홈":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        run_home()
        st.markdown('</div>', unsafe_allow_html=True)

elif menu == "🔍 나는 무슨 관상일까?":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        run_eda()
        st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📊 어떤 관상이 많을까?":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        run_ml()
        st.markdown('</div>', unsafe_allow_html=True)
