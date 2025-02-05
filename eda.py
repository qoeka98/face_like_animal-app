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


def run_eda():
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
                
                # ✅ df 변수를 전역 변수로 사용하도록 수정
                global df
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(data_file, index=False)  # ✅ CSV 파일 업데이트
                
                # ✅ 시각화
                fig = px.pie(names=[x[0] for x in sorted_results], values=[x[1] for x in sorted_results],
                             title="예측 확률 (%)", hole=0.3)
                st.subheader("📊 예측 결과")
                st.plotly_chart(fig, use_container_width=True)
                
                # ✅ 결과 출력
                st.success(f'🎉 당신은 **[{top_class}]** 상입니다! ({top_confidence:.1f}% 확률)')
            
            except Exception as e:
                st.error(f"이미지 처리 중 오류 발생: {e}")
