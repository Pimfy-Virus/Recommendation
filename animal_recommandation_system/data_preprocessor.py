import pandas as pd
import numpy as np
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

class AnimalDataProcessorForGPT:
    def __init__(self):
        self.processed_df = None
        
    def load_data(self, file_path):
        """CSV 파일 로드"""
        print("📂 데이터 로딩 중...")
        df = pd.read_csv(file_path)
        print(f"✅ 데이터 로딩 완료: {len(df)}개 레코드, {len(df.columns)}개 컬럼")
        return df
    
    def basic_cleaning(self, df):
        """기본적인 데이터 정리 (GPT 임베딩용)"""
        print("\n🔧 기본 데이터 정리 중...")
        
        df_clean = df.copy()
        
        # 몸무게 데이터 타입 정리만 수행 (이상치 수정 X)
        def clean_weight(weight):
            try:
                weight_float = float(weight)
                # 음수나 0은 무효 처리
                if weight_float <= 0:
                    return np.nan
                return weight_float
            except:
                return np.nan
        
        df_clean['addinfo07'] = df_clean['addinfo07'].apply(clean_weight)
        
        # 이상치 현황만 보고
        outliers = df_clean[df_clean['addinfo07'] > 100]
        print(f"   - 100kg 이상 몸무게: {len(outliers)}개 (원본 유지)")
        
        # 음수나 0 몸무게 처리
        negative_weights = (df['addinfo07'].astype(str).str.contains('-', na=False)).sum()
        print(f"   - 비정상 몸무게 {negative_weights}개 제거")
        
        return df_clean
    
    def handle_missing_values(self, df):
        """결측값 처리 (GPT가 이해할 수 있는 형태로)"""
        print("\n🔄 결측값 처리 중...")
        
        df_clean = df.copy()
        
        # GPT가 이해할 수 있는 기본값으로 설정
        missing_fields = {
            'addinfo01': '이름미정',           # 동물 이름
            'addinfo02': '구조위치미정',       # 구조 위치
            'addinfo03': '성별미정',           # 성별
            'addinfo04': '중성화여부미정',     # 중성화 여부
            'addinfo05': '나이정보없음',       # 나이
            'addinfo08': '',                   # 성격 해시태그 (빈 문자열)
            'addinfo09': '',                   # 구조 스토리 (빈 문자열)
            'addinfo10': '',                   # 성격 특성 (빈 문자열)
            'addinfo11': '',                   # 추가 정보 (빈 문자열)
            'addinfo16': '',                   # 특별 요구사항 (빈 문자열)
            'addinfo20': '',                   # 일상 관리 정보 (빈 문자열)
            'state': '입양상태미정',           # 입양 상태
            'kind': '임보종류미정'             # 임보 종류
        }
        
        for field, default_value in missing_fields.items():
            if field in df_clean.columns:
                missing_count = df_clean[field].isna().sum()
                df_clean[field] = df_clean[field].fillna(default_value)
                if missing_count > 0:
                    print(f"   - {field}: {missing_count}개 결측값 → '{default_value}'")
        
        # 몸무게 결측값은 그대로 유지 (자연어로 처리)
        weight_missing = df_clean['addinfo07'].isna().sum()
        if weight_missing > 0:
            print(f"   - 몸무게 결측값 {weight_missing}개 (자연어 텍스트에서 '몸무게 정보 없음'으로 처리 예정)")
        
        return df_clean
    
    def create_size_category(self, weight):
        """몸무게를 자연어 크기 표현으로 변환"""
        if pd.isna(weight):
            return "몸무게 정보 없음"
        
        try:
            weight = float(weight)
            if weight < 7:
                return "소형 (7kg 미만)"
            elif weight < 20:
                return "중형 (7-20kg)"
            else:
                return "대형 (20kg 이상)"
        except:
            return "몸무게 정보 없음"
    
    def create_age_description(self, age_info):
        """나이 정보를 자연어로 변환"""
        if pd.isna(age_info) or age_info == '나이정보없음':
            return "나이 정보 없음"
        
        age_str = str(age_info).lower()
        
        # 연도 기반 분류
        if any(year in age_str for year in ['2024', '2023']):
            return "어린 동물 (1-2세 추정)"
        elif any(year in age_str for year in ['2022', '2021', '2020']):
            return "젊은 성체 (3-5세 추정)"
        elif any(year in age_str for year in ['2019', '2018', '2017']):
            return "중년 동물 (6-8세 추정)"
        elif any(year in age_str for year in ['2016', '2015', '2014', '2013']):
            return "고령 동물 (9세 이상 추정)"
        else:
            return f"나이 관련 정보: {age_info}"
    
    def clean_text_for_gpt(self, text):
        """GPT가 이해하기 쉽도록 텍스트 정리"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 1. 해시태그를 자연어로 변환 (#애교쟁이 → 애교쟁이)
        # GPT가 해시태그도 충분히 이해하므로 단순 변환만
        text = re.sub(r'#([가-힣a-zA-Z0-9]+)', r'\1', text)
        
        # 2. 기본적인 정리만 수행
        text = re.sub(r'&apos;', "'", text)  # HTML 엔티티
        text = re.sub(r'\r\n', ' ', text)    # 줄바꿈을 공백으로
        text = re.sub(r'\n', ' ', text)      # 줄바꿈을 공백으로
        text = re.sub(r'\s+', ' ', text)     # 연속 공백 제거
        
        return text.strip()
    
    def create_comprehensive_description(self, row):
        """각 동물의 종합적인 자연어 설명 생성"""
        
        # 기본 정보 수집
        name = row.get('addinfo01', '이름미정')
        gender = row.get('addinfo03', '성별미정')
        weight = row.get('addinfo07')
        age_info = row.get('addinfo05', '나이정보없음')
        neuter = row.get('addinfo04', '중성화여부미정')
        state = row.get('state', '입양상태미정')
        kind = row.get('kind', '임보종류미정')
        
        # 크기와 나이 설명 생성
        size_desc = self.create_size_category(weight)
        age_desc = self.create_age_description(age_info)
        
        # 텍스트 필드들 정리
        personality_tags = self.clean_text_for_gpt(row.get('addinfo08', ''))
        rescue_story = self.clean_text_for_gpt(row.get('addinfo09', ''))
        personality_desc = self.clean_text_for_gpt(row.get('addinfo10', ''))
        additional_info = self.clean_text_for_gpt(row.get('addinfo11', ''))
        special_needs = self.clean_text_for_gpt(row.get('addinfo16', ''))
        daily_care = self.clean_text_for_gpt(row.get('addinfo20', ''))
        health_info = self.clean_text_for_gpt(row.get('addinfo19', ''))
        
        # 자연어 설명 구성
        description_parts = []
        
        # 1. 기본 소개
        intro = f"{name}는 {gender}이며, {size_desc}에 해당합니다."
        if age_desc != "나이 정보 없음":
            intro += f" {age_desc}이고"
        if neuter and neuter != '중성화여부미정':
            intro += f" {neuter} 상태입니다."
        else:
            intro += "입니다."
        description_parts.append(intro)
        
        # 2. 현재 상태
        if state != '입양상태미정' or kind != '임보종류미정':
            status = f"현재 {state} 상태이며 {kind}로 분류됩니다."
            description_parts.append(status)
        
        # 3. 성격 특징
        if personality_tags:
            description_parts.append(personality_tags)
        
        # 4. 성격 상세 설명
        if personality_desc:
            description_parts.append(f"성격 상세: {personality_desc}")
        
        # 5. 구조 배경
        if rescue_story:
            description_parts.append(f"구조 배경: {rescue_story}")
        
        # 6. 건강 정보
        if health_info:
            description_parts.append(f"건강 상태: {health_info}")
        
        # 7. 특별 요구사항
        if special_needs:
            description_parts.append(f"특별 요구사항: {special_needs}")
        
        # 8. 일상 관리
        if daily_care:
            description_parts.append(f"일상 관리: {daily_care}")
        
        # 9. 추가 정보
        if additional_info:
            description_parts.append(f"추가 정보: {additional_info}")
        
        # 최종 설명 결합
        final_description = ' '.join(description_parts)
        
        return final_description
    
    def process_for_gpt_embedding(self, file_path):
        """GPT 임베딩을 위한 전체 전처리 파이프라인"""
        print("🤖 GPT 임베딩용 데이터 전처리 시작")
        print("=" * 50)
        
        # 1. 데이터 로딩
        df = self.load_data(file_path)
        
        # 2. 기본 정리
        df = self.basic_cleaning(df)
        
        # 3. 결측값 처리
        df = self.handle_missing_values(df)
        
        # 4. 자연어 설명 생성
        print("\n📝 자연어 설명 생성 중...")
        descriptions = []
        
        for idx, row in df.iterrows():
            description = self.create_comprehensive_description(row)
            descriptions.append(description)
            
            # 진행률 표시
            if (idx + 1) % 500 == 0:
                print(f"   진행률: {idx + 1}/{len(df)} ({((idx + 1)/len(df)*100):.1f}%)")
        
        # 5. 설명 텍스트를 데이터프레임에 추가
        df['gpt_description'] = descriptions
        
        # 처리된 데이터 저장
        self.processed_df = df
        
        print(f"\n✅ GPT 임베딩용 전처리 완료!")
        print(f"   - 최종 데이터 크기: {len(df)}개")
        
        # 텍스트 길이 통계
        desc_lengths = [len(desc) for desc in descriptions]
        print(f"   - 평균 설명 길이: {np.mean(desc_lengths):.0f}자")
        print(f"   - 최대 설명 길이: {max(desc_lengths)}자")
        print(f"   - 최소 설명 길이: {min(desc_lengths)}자")
        
        return df, descriptions
    
    def save_processed_data(self, output_path="gpt_preprocessed_data.pkl"):
        """전처리된 데이터 저장"""
        print(f"\n💾 전처리된 데이터 저장 중: {output_path}")
        
        processed_data = {
            'dataframe': self.processed_df,
            'descriptions': self.processed_df['gpt_description'].tolist(),
            'metadata': {
                'total_records': len(self.processed_df),
                'preprocessing_type': 'GPT_embedding_optimized',
                'columns': list(self.processed_df.columns)
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"✅ 저장 완료: {output_path}")
    
    def sample_results(self, n=3):
        """처리 결과 샘플 확인"""
        print(f"\n🔍 GPT 전처리 결과 샘플 ({n}개)")
        print("-" * 80)
        
        for i in range(min(n, len(self.processed_df))):
            row = self.processed_df.iloc[i]
            
            print(f"\n【동물 {i+1}: {row['addinfo01']}】")
            print(f"생성된 자연어 설명:")
            print(f"   {row['gpt_description'][:300]}...")
            print(f"   (총 {len(row['gpt_description'])}자)")

# 사용 예시
if __name__ == "__main__":
    # GPT용 전처리 객체 생성
    gpt_processor = AnimalDataProcessorForGPT()
    
    # 전체 파이프라인 실행
    df, descriptions = gpt_processor.process_for_gpt_embedding('homeprotection_data.csv')
    
    # 결과 샘플 확인
    gpt_processor.sample_results(3)
    
    # 전처리된 데이터 저장
    gpt_processor.save_processed_data()
    
    print("\n🎯 다음 단계: GPT 임베딩 생성")
    print("   → GPTEmbeddingProcessor로 임베딩 벡터 생성")