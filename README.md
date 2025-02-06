# face_like_animal-app

### 📌 **Face Analysis AI App**  
🚀 **AI를 활용한 얼굴 분석! 당신의 관상은?**  

이 프로젝트는 **Streamlit과 Keras를 활용하여 사용자의 얼굴을 분석하고, 닮은 동물을 찾아주는 AI 웹 애플리케이션**입니다.  
사용자가 사진을 업로드하면 AI가 예측을 수행하고, 결과를 데이터로 저장하며, 전체 관상 데이터를 시각화할 수 있습니다.

---

## 🌟 **프로젝트 개요**
✔️ **사진 업로드**: 사용자가 자신의 얼굴 사진을 업로드하면 AI가 분석  
✔️ **AI 모델 예측**: Keras 모델을 활용해 닮은 동물 예측  
✔️ **데이터 저장**: 예측 결과를 CSV 파일에 저장하여 유저 데이터 축적  
✔️ **데이터 시각화**: 전체 사용자 데이터 기반 관상 트렌드 분석  

---

## 🎭 **주요 기능**
### 🔍 1. **관상 분석 (Face Analysis)**
- 사용자가 사진을 업로드하면 AI가 닮은 동물을 예측  
- 예측된 결과와 확률을 표시  
- 데이터를 지속적으로 축적하여 더욱 정밀한 분석 가능  

### 📊 2. **통계 및 시각화 (Data Visualization)**
- 사용자가 많아질수록 **관상 유형별 분석** 가능  
- **누적된 데이터**를 활용해 **각 관상의 개수 및 확률 분포를 그래프**로 제공  
- 📊 **Plotly를 이용한 시각화 지원**  

### 🏠 3. **홈 화면 (Project Overview)**
- **앱 개발 과정 및 AI 모델 학습 과정 소개**  
- **Teachable Machine을 활용한 학습 과정 설명**  
- **Streamlit 배포 과정 및 기술 스택 소개**  

---

## 🛠 **기술 스택**
| 기술 | 설명 |
|------|------|
| **Streamlit** | AI 모델 배포 및 웹 UI 개발 |
| **Keras (TensorFlow)** | 이미지 분류 모델 활용 |
| **Pandas** | 예측 데이터 저장 및 관리 |
| **Plotly** | 데이터 시각화 |
| **PIL (Pillow)** | 이미지 전처리 |
| **Numpy** | 데이터 변환 및 모델 입력 처리 |

---

## 🔧 **설치 및 실행 방법**
### 1️⃣ **필요한 패키지 설치**
```bash
pip install -r requirements.txt
```

### 2️⃣ **앱 실행**
```bash
streamlit run app.py
```

---

## 🚀 **Streamlit 배포 과정**
1. 로컬에서 Streamlit을 활용해 **테스트 실행**
2. `requirements.txt` 파일을 생성하여 **환경 구성**
3. **GitHub에 업로드 후, Streamlit Share 또는 다른 서버를 통해 배포**

---

## 📂 **프로젝트 구조**
```
📁 face-analysis-app/
│── 📄 app.py                # 메인 실행 파일 (Streamlit)
│── 📂 model/                # AI 모델 및 라벨 저장
│   │── keras_model.h5       # 학습된 AI 모델
│   │── labels.txt           # 분류할 클래스 라벨
│── 📂 pages/                # Streamlit 서브페이지
│   │── home.py              # 홈 화면
│   │── eda.py               # 관상 분석 기능
│   │── ml.py                # 데이터 시각화
│── 📄 data.csv              # 사용자 예측 데이터 저장 파일
│── 📄 requirements.txt      # 필요 패키지 리스트
│── 📄 README.md             # 프로젝트 설명 문서
```

---


