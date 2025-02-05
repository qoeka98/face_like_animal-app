import os
import numpy as np
import pandas as pd  # ✅ 데이터 저장을 위한 pandas 추가
import streamlit as st
from keras.models import load_model  
from PIL import Image, ImageOps
import plotly.express as px  
import time

from eda import run_eda
from home import run_home
from ml import run_ml  



# 👉 사이드바 메뉴
menu = st.sidebar.radio("메뉴 선택", ["🏠 홈", "🔍 나는 무슨 관상일까?", "📊 어떤 관상이 많을까?"])

if menu == "🏠 홈":
    run_home()

elif menu == "🔍 나는 무슨 관상일까?":
    run_eda()


elif menu == "📊 어떤 관상이 많을까?":
    run_ml()