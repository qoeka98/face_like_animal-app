import os
import numpy as np
import pandas as pd  # ✅ 데이터 저장을 위한 pandas 추가
import streamlit as st
from keras.models import load_model  
from PIL import Image, ImageOps
import plotly.express as px  
import time



def run_home():
    st.title("🎭 AI 얼굴 분석")
    st.markdown("이 앱은 인공지능을 이용하여 얼굴 관상을 분석하는 앱입니다.")
    st.image('image/입춘.png', caption="AI 기반 얼굴 분석", use_container_width=True)
    if st.button('어떻게 만들어진 앱일까?'):
        
        st.markdown("""
## 📌 티쳐블머신을 활용한 관상 분석 모델

""")
        st.image('image/789.png')
        st.divider()
    
        st.subheader('티쳐블머신 학습')
        st.markdown('''티쳐블 머신을 활용하여 관상 분석 모델을 학습시켰다.  
적절한 학습량을 설정하기 위해 실험을 진행한 결과, **에포크 100은 학습이 충분하지 않았고, 200은 과적합으로 인해 오류가 발생**하였다. 따라서 최적의 학습량으로 **에포크 150을 설정**하였다.  

학습 데이터는 **네이버 이미지와 구글 이미지에서 최소 45~50장씩 수집**하여 구성하였으며, 이를 **트레이닝(training)과 테스트(test) 데이터셋으로 분리하여 저장**하였다. 이후 해당 이미지를 활용하여 모델을 학습시켰다.  

완성된 모델은 **티쳐블 머신에서 내보낸 후, VS Code를 이용하여 활용할 수 있도록 구성**하였다.''')  
        st.write('')
        st.divider()
        st.write('')
        st.subheader('배포과정')
        st.markdown('''배포 과정에서는 **Streamlit을 사용하여 웹 애플리케이션 형태로 구현**하였다. 초기에는 **로컬 환경에서 테스트를 진행한 후, `requirements.txt` 파일을 생성하여 외부 환경에서도 실행 가능하도록 설정**하였다. 이를 통해 모델을 성공적으로 배포할 수 있었다.    ''')
        st.write('')
        st.divider()
        st.write('')
        st.subheader('앱의 장점')
        st.markdown("""📌이 앱의 주요 장점은 **사용자가 자신의 얼굴과 닮은 동물을 알 수 있다는 점**이며, **유저들이 사용할수록 데이터가 축적**되어 더 정확한 분석이 가능하다는 것입니다.  

특히, **축적된 데이터를 통해 사용자는 자신의 관상이 전체 데이터에서 어떤 평균적 특징을 가지는지 확인할 수 있습니다**. 이는 단순한 AI 분석을 넘어, 집단적인 데이터를 기반으로 한 **관상 평균 지점(트렌드) 분석**까지 가능하게 합니다.  

또한, 개발 과정에서 **복잡한 데이터베이스(DB) 시스템을 사용하지 않고**, **로컬 CSV 파일을 활용하는 방식을 선택**하였습니다.  
이를 통해 **구현이 간편하고, 스트림릿 환경에서 빠르게 데이터를 저장하고 불러올 수 있도록 최적화**하였습니다. """)
        st.write('')
        st.divider()
        st.write('')
        st.markdown('''
                    사진출쳐:https://www.google.com/search?sca_esv=b71b87a039ad9bf1&sxsrf=AHTn8zpmna7qllmBxI166z1aeHuUw5mk4A:1738723799613&q=%EA%B0%95%EC%95%84%EC%A7%80%EC%83%81%EC%97%B0%EC%98%88%EC%9D%B8&udm=2&fbs=ABzOT_CWdhQLP1FcmU5B0fn3xuWpA-dk4wpBWOGsoR7DG5zJBpwxALD7bRaeOIZxqOFEngzB_O_LYSS4XXpaWwzVPCpGAm7zOmiX81RBvM6Jl5WVFTU8lMVsZqZi3IU8-OUPC-849zpywWzyFJoPFXz4gPBunfYrO5qbT5mEc2e_hxvEGkF—H1zOWJDXBbLgIqxF_SNtk6Y&sa=X&ved=2ahUKEwir1MDzwquLAxWqna8BHUQmLgwQtKgLegQIEBAB&biw=1745&bih=828&dpr=1.1

사진출쳐:
https://search.naver.com/search.naver?nso=so%3Ar%2Cp%3Aall&query=%EA%B3%A0%EC%96%91%EC%9D%B4%EC%83%81+%EC%97%B0%EC%98%88%EC%9D%B8&sm=tab_nmr&where=image
''')
        st.write('')
        st.divider()
        st.write('')
        st.subheader('개발자 깃허브')
        st.markdown('https://github.com/qoeka98/face_like_animal-app')
        st.write('')
        

    