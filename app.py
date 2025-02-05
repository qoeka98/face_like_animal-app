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
                
                # ✅ 관상 설명 추가
                description = ""
                if top_class == "강아지":
                    description = "순한 인상을 갖고 있습니다."
                elif top_class == "고양이":
                    description = "고양이와 같은 매력적인 인상을 갖고 있습니다."
                elif top_class == "돼지":
                    description = "복스러운 인상을 갖고 있습니다."
                
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
                
                # ✅ 결과 및 설명 출력
                st.success(f'🎉 당신은 **[{top_class}]** 상입니다! ({top_confidence:.1f}% 확률) {description}')
            
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
