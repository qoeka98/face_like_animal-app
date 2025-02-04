import os
import numpy as np
import pandas as pd  # ✅ 데이터 저장을 위한 pandas 추가
import streamlit as st
from keras.models import load_model  
from PIL import Image, ImageOps
import plotly.express as px  
import time  

# 👉 사이드바 메뉴
st.markdown(
    """
    <style>
        /* 전체 페이지 배경색 변경 */
        body, [data-testid="stAppViewContainer"] {
            background-color: #E6E6FA;  /* 파스텔톤 연보라색 */
        }
        
        /* 사이드바 배경색 변경 */
        [data-testid="stSidebar"] {
            background-color: #FFD1DC  /* 연한 연보라색 */
        }
        
        /* 카드(컨테이너) 스타일 */
        .stApp {
            background-color: #E6E6FA;  /* 파스텔 연보라 */
        }
    </style>
    """,
    unsafe_allow_html=True
)


menu = st.sidebar.radio("메뉴 선택", ["🏠 홈", "🔍 나는 무슨 관상일까?", "📊 어떤 관상이 많을까?"])

# ✅ 세션 상태에서 데이터 저장을 위한 초기화
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame(columns=["날짜", "예측된 관상", "확률"])

if menu == "🏠 홈":
    st.title("🎭 AI 얼굴 분석")
    st.markdown("이 앱은 인공지능을 이용하여 얼굴 관상을 분석하는 앱입니다.")
    st.image('image/입춘.png', caption="AI 기반 얼굴 분석", use_container_width=True)
    

elif menu == "🔍 나는 무슨 관상일까?":
    st.title("🔍 나는 무슨 관상일까?")
    st.info("사진을 넣어주시면, AI가 분석하여 당신의 관상을 알려드립니다.")

    file = st.file_uploader('얼굴 사진을 업로드하세요', type=['jpg', 'png', 'jpeg'])

    if file is not None:
        # 이미지 로드
        image = Image.open(file)

        # 📌 연핑크 테두리 추가
        border_color = (255, 182, 193)  
        border_size = 10  
        image_with_border = ImageOps.expand(image, border=border_size, fill=border_color)

        # 중앙 정렬하여 이미지 표시
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.image(image_with_border, caption="내 사진", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button('결과보기'):
            with st.spinner('AI가 분석 중입니다...'):
                time.sleep(2)

            model_path = "model/keras_model.h5"
            if not os.path.exists(model_path):
                st.error(f"모델 파일이 존재하지 않습니다: {model_path}")
                st.stop()

            try:
                model = load_model(model_path)
                labels_path = "model/labels.txt"
                if not os.path.exists(labels_path):
                    st.error(f"라벨 파일이 존재하지 않습니다: {labels_path}")
                    st.stop()

                with open(labels_path, 'r', encoding='utf-8') as f:
                    class_names = f.read().splitlines()
                
            except Exception as e:
                st.error(f"모델 또는 라벨 파일을 불러오는 중 오류 발생: {e}")
                st.stop()

            try:
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                prediction = model.predict(data)[0]  
                
                sorted_indices = np.argsort(prediction)[::-1]  
                sorted_results = [(class_names[i], prediction[i] * 100) for i in sorted_indices]

                top_class, top_confidence = sorted_results[0]

                class_names_sorted = [item[0] for item in sorted_results]  
                confidences_sorted = [item[1] for item in sorted_results]  

                fig = px.pie(
                    names=class_names_sorted,
                    values=confidences_sorted,
                    title="예측 확률 (%)",
                    hole=0.3,  
                )
                fig.update_traces(textinfo='percent+label')

                st.subheader("📊 예측 결과 (전체)")
                st.plotly_chart(fig, use_container_width=True)

                st.success(f'🎉 당신은 **[{top_class}]** 상입니다! ({top_confidence:.1f}% 확률)')

                # ✅ 예측 결과를 데이터베이스(세션 상태)에 저장
                new_data = pd.DataFrame(
                    [[time.strftime("%Y-%m-%d %H:%M:%S"), top_class, top_confidence]],
                    columns=["날짜", "예측된 관상", "확률"]
                )
                st.session_state["data"] = pd.concat([st.session_state["data"], new_data], ignore_index=True)

            except Exception as e:
                st.error(f"이미지 처리 또는 예측 중 오류 발생: {e}")

elif menu == "📊 어떤 관상이 많을까?":
    st.title("📊 어떤 관상이 많을까?")
    st.write("이전 분석 결과를 기반으로 데이터를 저장하고, 그래프로 시각화합니다.")

    if not st.session_state["data"].empty:
        # ✅ 표 출력
        st.subheader("📋 누적된 관상 분석 데이터")
        st.dataframe(st.session_state["data"])

        # ✅ 관상별 비율을 그래프로 시각화
        st.subheader("📊 관상 비율 분석")

        # 클래스별 개수 집계
        count_data = st.session_state["data"]["예측된 관상"].value_counts().reset_index()
        count_data.columns = ["관상", "개수"]

        fig_bar = px.bar(
            count_data, 
            x="관상", 
            y="개수", 
            title="각 관상의 분석된 개수", 
            text="개수",
            color="관상"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ✅ 관상별 평균 확률을 그래프로 시각화
        avg_confidence = st.session_state["data"].groupby("예측된 관상")["확률"].mean().reset_index()
        fig_pie = px.pie(
            avg_confidence, 
            names="예측된 관상", 
            values="확률", 
            title="관상별 평균 확률 (%)",
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.warning("📢 아직 저장된 데이터가 없습니다. 먼저 관상 분석을 진행하세요!")
