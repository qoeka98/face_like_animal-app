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
    df = pd.read_csv(data_file)
else:
    df = pd.DataFrame(columns=["날짜", "예측된 관상", "확률"])

# 👉 사이드바 메뉴
menu = st.sidebar.radio("메뉴 선택", ["🏠 홈", "🔍 나는 무슨 관상일까?", "📊 어떤 관상이 많을까?"])

if menu == "🏠 홈":
    st.title("🎭 AI 얼굴 분석")
    st.markdown("이 앱은 인공지능을 이용하여 얼굴 관상을 분석하는 앱입니다.")
    st.image('image/입춘.png', caption="AI 기반 얼굴 분석", use_container_width=True)
    if st.button('어떻게 만들어진 앱일까?'):
        
        st.markdown("""
## 📌 티쳐블머신을 활용한 관상 분석 모델

""")
        st.image('image/789.png')
        st.markdown('''티쳐블 머신을 활용하여 관상 분석 모델을 학습시켰다.  
적절한 학습량을 설정하기 위해 실험을 진행한 결과, **에포크 100은 학습이 충분하지 않았고, 200은 과적합으로 인해 오류가 발생**하였다. 따라서 최적의 학습량으로 **에포크 150을 설정**하였다.  

학습 데이터는 **네이버 이미지와 구글 이미지에서 최소 45~50장씩 수집**하여 구성하였으며, 이를 **트레이닝(training)과 테스트(test) 데이터셋으로 분리하여 저장**하였다. 이후 해당 이미지를 활용하여 모델을 학습시켰다.  

완성된 모델은 **티쳐블 머신에서 내보낸 후, VS Code를 이용하여 활용할 수 있도록 구성**하였다.  

배포 과정에서는 **Streamlit을 사용하여 웹 애플리케이션 형태로 구현**하였다. 초기에는 **로컬 환경에서 테스트를 진행한 후, `requirements.txt` 파일을 생성하여 외부 환경에서도 실행 가능하도록 설정**하였다. 이를 통해 모델을 성공적으로 배포할 수 있었다.    ''')
        st.markdown("""📌이 앱의 주요 장점은 **사용자가 자신의 얼굴과 닮은 동물을 알 수 있다는 점**이며, **유저들이 사용할수록 데이터가 축적**되어 더 정확한 분석이 가능하다는 것입니다.  

특히, **축적된 데이터를 통해 사용자는 자신의 관상이 전체 데이터에서 어떤 평균적 특징을 가지는지 확인할 수 있습니다**. 이는 단순한 AI 분석을 넘어, 집단적인 데이터를 기반으로 한 **관상 평균 지점(트렌드) 분석**까지 가능하게 합니다.  

또한, 개발 과정에서 **복잡한 데이터베이스(DB) 시스템을 사용하지 않고**, **로컬 CSV 파일을 활용하는 방식을 선택**하였습니다.  
이를 통해 **구현이 간편하고, 스트림릿 환경에서 빠르게 데이터를 저장하고 불러올 수 있도록 최적화**하였습니다. """)
        st.markdown('''
                    
                    
                    
                    
                    사진출쳐:https://www.google.com/search?sca_esv=b71b87a039ad9bf1&sxsrf=AHTn8zpmna7qllmBxI166z1aeHuUw5mk4A:1738723799613&q=%EA%B0%95%EC%95%84%EC%A7%80%EC%83%81%EC%97%B0%EC%98%88%EC%9D%B8&udm=2&fbs=ABzOT_CWdhQLP1FcmU5B0fn3xuWpA-dk4wpBWOGsoR7DG5zJBpwxALD7bRaeOIZxqOFEngzB_O_LYSS4XXpaWwzVPCpGAm7zOmiX81RBvM6Jl5WVFTU8lMVsZqZi3IU8-OUPC-849zpywWzyFJoPFXz4gPBunfYrO5qbT5mEc2e_hxvEGkF—H1zOWJDXBbLgIqxF_SNtk6Y&sa=X&ved=2ahUKEwir1MDzwquLAxWqna8BHUQmLgwQtKgLegQIEBAB&biw=1745&bih=828&dpr=1.1


사진출쳐:
https://search.naver.com/search.naver?nso=so%3Ar%2Cp%3Aall&query=%EA%B3%A0%EC%96%91%EC%9D%B4%EC%83%81+%EC%97%B0%EC%98%88%EC%9D%B8&sm=tab_nmr&where=image
''')

    

elif menu == "🔍 나는 무슨 관상일까?":
    st.title("🔍 나는 무슨 관상일까?")
    st.info("사진을 넣어주시면, AI가 분석하여 당신의 관상을 알려드립니다.")

    file = st.file_uploader('얼굴 사진을 업로드하세요', type=['jpg', 'png', 'jpeg'])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="내 사진", width=300)

        if st.button('결과보기'):
            with st.spinner('AI가 분석 중입니다...'):
                time.sleep(2)

            model_path = "model/keras_model.h5"
            labels_path = "model/labels.txt"
            
            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                st.error("필요한 모델 파일이 존재하지 않습니다.")
                st.stop()
            
            try:
                model = load_model(model_path)
                with open(labels_path, 'r', encoding='utf-8') as f:
                    class_names = f.read().splitlines()
            except Exception as e:
                st.error(f"모델 로딩 중 오류 발생: {e}")
                st.stop()
            
            try:
                image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                prediction = model.predict(data)[0]  
                sorted_indices = np.argsort(prediction)[::-1]  
                sorted_results = [(class_names[i], prediction[i] * 100) for i in sorted_indices]

                top_class, top_confidence = sorted_results[0]
                
              
                
                # ✅ CSV에 데이터 저장
                new_data = pd.DataFrame([[time.strftime("%Y-%m-%d %H:%M:%S"), top_class, top_confidence]],
                                        columns=["날짜", "예측된 관상", "확률"])
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(data_file, index=False)  # ✅ CSV 파일 업데이트
                
                # ✅ 시각화
                fig = px.pie(names=[x[0] for x in sorted_results], values=[x[1] for x in sorted_results],
                             title="예측 확률 (%)", hole=0.3)
                st.subheader("📊 예측 결과")
                st.plotly_chart(fig, use_container_width=True)
                
              
                st.success(f'🎉 당신은 **[{top_class}]** 상입니다! ({top_confidence:.1f}% 확률)')
              
            
            except Exception as e:
                st.error(f"이미지 처리 중 오류 발생: {e}")


elif menu == "📊 어떤 관상이 많을까?":
    st.title("📊 어떤 관상이 많을까?")
    
    if df.empty:
        st.warning("📢 아직 저장된 데이터가 없습니다. 먼저 관상 분석을 진행하세요!")
    else:
        # ✅ 누적 데이터 표시
        st.subheader("📋 분석 데이터")
        st.dataframe(df)
        
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