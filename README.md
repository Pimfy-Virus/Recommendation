## 🐾 임시보호 동물 추천 시스템



이 프로젝트는 Streamlit 기반의 웹 애플리케이션으로, 사용자 맞춤 조건에 따라 임시보호 중인 동물을 추천합니다.



---

### 📁 프로젝트 구조

```
├── data/
│   └── pimfyvirus_dog_data.csv       # 앱에서 사용하는 임시보호 동물 데이터 CSV 파일
├── streamlit_app.py                  # Streamlit 기반 웹 애플리케이션 메인 실행 파일
├── data_preprocessor.py              # 데이터 불러오기 및 전처리 클래스 정의
├── animal_filter.py                  # 조건 필터링 및 추천 알고리즘 구현 클래스
├── db_test.py                        # 데이터베이스 연결 테스트 또는 관련 기능 샘플 코드
├── main.py                           # 로컬 테스트용 메인 실행 스크립트 (CLI 또는 개발 중 디버깅용)
├── config.py                         # 설정 값(경로, 상수 등) 모듈화한 설정 파일
├── README.md                         # 프로젝트 설명 및 실행 가이드 문서
```

---

### ⚙️ 설치 방법

1. Python 3.8+ 권장
2. 가상환경 설정 (선택)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

> `requirements.txt`가 없다면 직접 설치:

```bash
pip install streamlit pandas
```

---

### 🚀 실행 방법

아래 명령어를 \*\*터미널(명령 프롬프트)\*\*에서 실행하세요:

```bash
streamlit run streamlit_app.py
```

앱이 실행되면 브라우저가 자동으로 열리며, 보통 아래 주소에서 접속됩니다:

```
http://localhost:8501
```

---

### 📌 주의 사항

* `data/pimfyvirus_dog_data` 파일이 존재해야 앱이 실행됩니다.
* 경로가 다르면 `streamlit_app.py`에서 직접 경로를 수정하세요:

```python
CSV_PATH = "data/animals.csv"
```

---

### 🧠 기능 요약

* 🔍 조건별 필터링: 지역, 성별, 나이, 몸무게, 중성화, 성격 태그 필터
* 🤖 스마트 추천: AI 기반 선호도 조건 추천
* 📊 데이터 요약: 동물 통계 시각화
