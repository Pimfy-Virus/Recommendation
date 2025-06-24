import streamlit as st
import pandas as pd
from data_preprocessor import DataPreprocessor
from animal_filter import AnimalFilter

# 페이지 설정
st.set_page_config(page_title="임시보호 동물 추천 시스템", layout="wide")

# ✅ CSV 경로 직접 지정
CSV_PATH = "/Users/sxxwings/git/pimfy/Recommendation/data/pimfyvirus_dog_data.csv"

# ✅ 데이터 로딩
preprocessor = DataPreprocessor()
processed_data = preprocessor.load_and_process(CSV_PATH)
metadata = preprocessor.get_metadata()

animal_filter = AnimalFilter()
animal_filter.set_animals(processed_data)

st.success("✅ 데이터가 자동으로 로딩되었습니다!")

# ✅ 탭 구성
tab1, tab2, tab3 = st.tabs(["🔍 조건별 필터링", "🤖 스마트 추천", "📊 데이터 요약"])

# 🔍 조건별 필터링 탭
with tab1:
    st.subheader("🔍 조건별 필터링")
    region = st.selectbox("📍 구조 지역 선택", [""] + list(metadata['regions']))
    gender = st.selectbox("⚥ 성별", ["", "male", "female"])
    age_range = st.slider("🎂 나이 범위", 0, 20, (0, 10))
    weight_range = st.slider("⚖️ 몸무게 범위 (kg)", 0.0, 50.0, (0.0, 20.0))
    neutered = st.radio("✂️ 중성화 여부", ["무관", "중성화 완료", "중성화 안함"], horizontal=True)
    hashtags = st.multiselect("🏷️ 성격 해시태그", metadata["all_hashtags"])

    if st.button("필터링 적용"):
        filters = {
            "region": [region] if region else None,
            "gender": [gender] if gender else None,
            "age_range": {"min": age_range[0], "max": age_range[1]},
            "weight_range": {"min": weight_range[0], "max": weight_range[1]},
            "neutered": None if neutered == "무관" else (neutered == "중성화 완료"),
            "hashtags": hashtags if hashtags else None
        }

        # None 제거
        filters = {k: v for k, v in filters.items() if v is not None}
        results = animal_filter.apply_filters(filters)

        st.write(f"✅ 조건에 맞는 동물 수: {len(results)}마리")
        st.dataframe(results.head(10))

# 🤖 AI 스마트 추천 탭
with tab2:
    st.subheader("🤖 AI 스마트 추천")

    st.markdown("##### 🎂 선호 나이")
    pref_age = st.slider("선호 나이 범위", 0, 20, (2, 5))
    accept_age = st.slider("허용 나이 범위", 0, 20, (1, 8))

    st.markdown("##### ⚖️ 선호 몸무게")
    pref_weight = st.slider("선호 몸무게 (kg)", 0.0, 50.0, (3.0, 10.0))
    
    st.markdown("##### 🏷️ 원하는 성격 해시태그")
    personality_traits = st.multiselect("선호 성격", metadata["all_hashtags"])

    st.markdown("##### 🐕 행동 특성")
    affection = st.slider("애정 표현", 1, 5, 4)
    friendly = st.slider("사람 친화성", 1, 5, 5)
    barking = st.slider("짖음 정도", 1, 5, 2)

    if st.button("추천 시작"):
        preferences = {
            "age_preference": {
                "preferred": {"min": pref_age[0], "max": pref_age[1]},
                "acceptable": {"min": accept_age[0], "max": accept_age[1]},
            },
            "size_preference": {
                "preferred": {"min": pref_weight[0], "max": pref_weight[1]},
                "acceptable": {"min": 0, "max": 50},
            },
            "personality_traits": personality_traits,
            "behavior_preferences": {
                "affection": {
                    "ideal": affection,
                    "acceptable": [max(1, affection - 1), affection, min(5, affection + 1)],
                },
                "human_friendly": {
                    "ideal": friendly,
                    "acceptable": [max(1, friendly - 1), friendly, min(5, friendly + 1)],
                },
                "barking": {
                    "ideal": barking,
                    "acceptable": [max(1, barking - 1), barking, min(5, barking + 1)],
                },
            },
            "weights": {
                "age": 1.5,
                "size": 1.2,
                "personality": 1.8,
                "behavior": 1.3,
            },
        }

        recommendations = animal_filter.apply_soft_filtering(preferences)
        st.write(f"✅ 추천 동물 수: {len(recommendations)}마리")
        st.dataframe(recommendations.head(10))

# 📊 데이터 요약 탭
with tab3:
    st.subheader("📊 데이터 요약")
    stats = preprocessor.get_statistics()

    st.markdown(f"**전체 동물 수:** {stats['total']:,}마리")
    st.markdown(f"**임보 가능:** {stats['available']:,}마리")
    st.markdown(f"**평균 나이:** {stats['average_age']}세")
    st.markdown(f"**평균 몸무게:** {stats['average_weight']}kg")

    st.markdown("#### 성별 분포")
    st.bar_chart(pd.Series(stats['gender_distribution']))

    st.markdown("#### 주요 구조 지역")
    top_regions = dict(list(stats['region_distribution'].items())[:10])
    st.bar_chart(pd.Series(top_regions))

    st.markdown("#### 임보 종류 분포")
    top_care_types = dict(list(stats['care_type_distribution'].items())[:10])
    st.bar_chart(pd.Series(top_care_types))
