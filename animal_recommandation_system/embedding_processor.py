import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GPTEmbeddingProcessor:
    def __init__(self, api_key=None, model="text-embedding-3-large"):
        """
        GPT 임베딩 기반 벡터화 프로세서
        """
        if api_key:
            openai.api_key = api_key
        
        self.model = model
        self.embeddings = None
        self.processed_df = None
        self.embedding_dim = 3072
        
        print(f"🤖 GPT 임베딩 프로세서 초기화")
        print(f"   - 모델: {model}")
        print(f"   - 임베딩 차원: {self.embedding_dim}")
    
    def create_embedding_text(self, row):
        """동물 정보를 임베딩용 자연어 텍스트로 변환"""
        
        # 기본 정보
        name = row.get('addinfo01', '이름미정')
        gender = row.get('addinfo03', '성별미정')
        weight = row.get('addinfo07', '몸무게미정')
        age = row.get('addinfo05', '나이미정')
        neuter = row.get('addinfo04', '중성화미정')
        state = row.get('state', '상태미정')
        kind = row.get('kind', '임보종류미정')
        
        # 성격 및 특성 정보
        personality_tags = row.get('addinfo08', '')
        personality_desc = row.get('addinfo10', '')
        rescue_story = row.get('addinfo09', '')
        additional_info = row.get('addinfo11', '')
        special_needs = row.get('addinfo16', '')
        daily_care = row.get('addinfo20', '')
        
        # 크기 분류
        try:
            weight_float = float(weight) if weight != '몸무게미정' else None
            if weight_float:
                if weight_float < 10:
                    size = "소형"
                elif weight_float < 25:
                    size = "중형"
                else:
                    size = "대형"
            else:
                size = "크기미정"
        except:
            size = "크기미정"
        
        # 해시태그 처리
        if personality_tags:
            # #애교쟁이#사람좋아 → 애교가 많고 사람을 좋아하는
            cleaned_tags = re.sub(r'#([가-힣a-zA-Z0-9]+)', r'\1', personality_tags)
            tag_list = [tag for tag in cleaned_tags.split('#') if tag.strip()]
            personality_text = ', '.join(tag_list) if tag_list else ''
        else:
            personality_text = ''
        
        # 자연어 형태로 구성
        text_parts = []
        
        # 기본 정보 문장
        basic_info = f"{name}는 {gender}이고 몸무게 {weight}kg인 {size} 동물입니다."
        text_parts.append(basic_info)
        
        # 상태 정보
        if state and kind:
            status_info = f"현재 {state} 상태이며 {kind}로 분류됩니다."
            text_parts.append(status_info)
        
        # 성격 특성
        if personality_text:
            personality_info = f"성격 특징은 {personality_text} 입니다."
            text_parts.append(personality_info)
        
        # 성격 상세 설명 (토큰 효율성 고려하여 적절히 조정)
        if personality_desc:
            desc_text = personality_desc.strip()
            if desc_text:
                text_parts.append(f"성격 설명: {desc_text}")
        
        # 구조 배경 (전체 내용 포함 - GPT가 중요한 부분 알아서 판단)
        if rescue_story:
            story_text = rescue_story.strip()
            if story_text:
                text_parts.append(f"구조 배경: {story_text}")
        
        # 특별 요구사항 (전체 포함)
        if special_needs:
            needs_text = special_needs.strip()
            if needs_text:
                text_parts.append(f"특별 요구사항: {needs_text}")
        
        # 일상 관리 (전체 포함)
        if daily_care:
            care_text = daily_care.strip()
            if care_text:
                text_parts.append(f"일상 관리: {care_text}")
        
        # 최종 텍스트 결합
        final_text = ' '.join(text_parts)
        
        max_length = 30000  
        if len(final_text) > max_length:
            # 중요도 순으로 텍스트 유지 (기본정보 > 성격 > 기타)
            essential_parts = text_parts[:4]  # 기본정보 + 성격 정보
            essential_text = ' '.join(essential_parts)
            
            if len(essential_text) <= max_length:
                final_text = essential_text
            else:
                final_text = essential_text[:max_length] + "..."
        
        return final_text
    
    def get_embedding(self, text, max_retries=3):
        """단일 텍스트의 임베딩 생성"""
        for attempt in range(max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                return response.data[0].embedding
            
            except openai.RateLimitError:
                print(f"   Rate limit reached, waiting {2**attempt} seconds...")
                time.sleep(2**attempt)
            except Exception as e:
                print(f"   Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"   Failed to get embedding after {max_retries} attempts")
                    return None
                time.sleep(1)
        
        return None
    
    def get_embeddings_batch(self, texts, batch_size=100):
        """배치로 임베딩 생성 (비용 및 속도 최적화)"""
        print(f"\n🚀 GPT 임베딩 생성 중 (배치 크기: {batch_size})")
        
        embeddings = []
        failed_indices = []
        
        # 배치 단위로 처리
        for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성"):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # 배치 요청
                response = openai.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                    encoding_format="float"
                )
                
                # 결과 저장
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # API 요청 제한 고려
                time.sleep(0.1)  # 100ms 대기
                
            except Exception as e:
                print(f"\n❌ 배치 {i//batch_size + 1} 실패: {str(e)}")
                
                # 개별 처리로 fallback
                print("   개별 처리로 재시도...")
                for j, text in enumerate(batch_texts):
                    embedding = self.get_embedding(text)
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        embeddings.append(np.zeros(self.embedding_dim))
                        failed_indices.append(i + j)
                
                time.sleep(1)  # 실패 시 더 긴 대기
        
        print(f"✅ 임베딩 생성 완료")
        print(f"   - 성공: {len(embeddings) - len(failed_indices)}개")
        print(f"   - 실패: {len(failed_indices)}개")
        
        return np.array(embeddings), failed_indices
    
    def process_animal_data(self, df):
        """동물 데이터 전체 처리"""
        print("🐕 동물 데이터 임베딩 처리 시작")
        print("=" * 50)
        
        # 1. 임베딩용 텍스트 생성
        print("📝 임베딩용 텍스트 생성 중...")
        embedding_texts = []
        
        for idx, row in df.iterrows():
            text = self.create_embedding_text(row)
            embedding_texts.append(text)
            
            # 진행률 표시
            if (idx + 1) % 500 == 0:
                print(f"   진행률: {idx + 1}/{len(df)} ({((idx + 1)/len(df)*100):.1f}%)")
        
        print(f"✅ 텍스트 생성 완료: {len(embedding_texts)}개")
        
        # 텍스트 길이 통계
        text_lengths = [len(text) for text in embedding_texts]
        print(f"   - 평균 길이: {np.mean(text_lengths):.0f}자")
        print(f"   - 최대 길이: {max(text_lengths)}자")
        
        # 2. GPT 임베딩 생성
        embeddings, failed_indices = self.get_embeddings_batch(embedding_texts)
        
        # 3. 결과 저장
        self.embeddings = embeddings
        self.processed_df = df.copy()
        self.processed_df['embedding_text'] = embedding_texts
        
        return embeddings, embedding_texts
    
    def process_user_query(self, user_input):
        """사용자 쿼리를 임베딩으로 변환"""
        
        # 사용자 입력을 자연어로 정제
        processed_query = self.preprocess_user_query(user_input)
        
        # 임베딩 생성
        query_embedding = self.get_embedding(processed_query)
        
        if query_embedding is None:
            print("❌ 사용자 쿼리 임베딩 생성 실패")
            return None
        
        return np.array(query_embedding)
    
    def preprocess_user_query(self, user_input):
        """사용자 입력을 기본 정제만 수행 (GPT가 의미를 알아서 이해)"""
        
        # 기본 정제만 수행
        query = str(user_input).strip()
        
        # 불필요한 특수문자 제거
        query = re.sub(r'[^\w\s가-힣.,!?]', ' ', query)
        
        # 연속 공백 제거
        query = re.sub(r'\s+', ' ', query).strip()
        
        # GPT가 알아서 이해하므로 별도 변환 불필요
        return query
    
    def extract_user_preferences(self, user_input):
        """사용자 입력에서 하드 필터 조건 추출"""
        preferences = {
            'size': None,
            'age': None,
            'gender': None,
            'personality_query': user_input  # 전체 쿼리는 성격 매칭용
        }
        
        # 크기 조건 추출
        if any(keyword in user_input for keyword in ['소형견', '소형', '작은']):
            preferences['size'] = '소형'
        elif any(keyword in user_input for keyword in ['중형견', '중형', '중간']):
            preferences['size'] = '중형'
        elif any(keyword in user_input for keyword in ['대형견', '대형', '큰']):
            preferences['size'] = '대형'
        
        # 나이 조건 추출
        if any(keyword in user_input for keyword in ['어린', '새끼', '젊은']):
            preferences['age'] = '어린'
        elif any(keyword in user_input for keyword in ['나이많은', '고령', '시니어']):
            preferences['age'] = '고령'
        
        # 성별 조건 추출
        if any(keyword in user_input for keyword in ['수컷', '남자', '남']):
            preferences['gender'] = '남'
        elif any(keyword in user_input for keyword in ['암컷', '여자', '여']):
            preferences['gender'] = '여'
        
        return preferences
    
    def apply_hard_filters(self, df, embeddings, preferences):
        """물리적 조건으로 먼저 필터링"""
        # 인덱스를 리셋해서 0부터 시작하도록 만들기
        df_reset = df.reset_index(drop=True)
        
        mask = pd.Series([True] * len(df_reset))
        
        print(f"🔍 하드 필터 적용 전: {len(df_reset)}마리")
        
        # 크기 필터
        if preferences['size']:
            print(f"   크기 조건: {preferences['size']}")
            
            # 몸무게 데이터 확인 및 정리
            weights = pd.to_numeric(df_reset['addinfo07'], errors='coerce')
            
            if preferences['size'] == '소형':
                weight_mask = weights < 7
                print(f"   소형 조건 (<7kg): {weight_mask.sum()}마리")
            elif preferences['size'] == '중형':
                weight_mask = (weights >= 7) & (weights < 20)
                print(f"   중형 조건 (7-20kg): {weight_mask.sum()}마리")
            elif preferences['size'] == '대형':
                weight_mask = weights >= 20
                print(f"   대형 조건 (≥20kg): {weight_mask.sum()}마리")
            
            mask = mask & weight_mask
            print(f"   크기 필터 후: {mask.sum()}마리")
            
        # 성별 필터
        if preferences['gender']:
            print(f"   성별 조건: {preferences['gender']}")
            gender_mask = df_reset['addinfo03'] == preferences['gender']
            mask = mask & gender_mask
            print(f"   성별 필터 후: {mask.sum()}마리")
        
        # 나이 필터 (연도 기반)
        if preferences['age']:
            print(f"   나이 조건: {preferences['age']}")
            if preferences['age'] == '어린':
                age_mask = df_reset['addinfo05'].astype(str).str.contains('2024|2023', na=False)
                print(f"   어린 동물 (2023-2024): {age_mask.sum()}마리")
            elif preferences['age'] == '고령':
                age_mask = df_reset['addinfo05'].astype(str).str.contains('2013|2014|2015|2016', na=False)
                print(f"   고령 동물 (2013-2016): {age_mask.sum()}마리")
            else:
                age_mask = pd.Series([True] * len(df_reset))
            
            mask = mask & age_mask
            print(f"   나이 필터 후: {mask.sum()}마리")
        
        # 필터링된 결과 반환 (인덱스도 함께 맞춰서)
        filtered_df = df_reset[mask].reset_index(drop=True)
        filtered_embeddings = embeddings[mask.values]  # numpy 배열 인덱싱
        
        print(f"🎯 최종 필터링 결과: {len(filtered_df)}마리")
        
        # 필터링 결과 샘플 확인
        if len(filtered_df) > 0:
            print("   필터링된 동물 샘플:")
            for i in range(min(3, len(filtered_df))):
                row = filtered_df.iloc[i]
                print(f"   - {row['addinfo01']}: {row['addinfo07']}kg, {row['addinfo03']}")
        
        return filtered_df, filtered_embeddings
    
    def find_similar_animals(self, user_query, top_k=5, available_only=True):
        """하드 필터 + 성격 유사도 매칭"""
        
        if self.embeddings is None:
            print("❌ 임베딩 데이터가 없습니다.")
            return []
        
        print(f"🔍 사용자 쿼리 처리: '{user_query}'")
        
        # 1. 사용자 선호도 추출
        preferences = self.extract_user_preferences(user_query)
        print(f"🎯 추출된 선호도: {preferences}")
        
        # 2. 입양 가능 동물 필터링
        if available_only:
            available_mask = self.processed_df['state'] == '임보가능'
            filtered_df = self.processed_df[available_mask].reset_index(drop=True)
            
            # numpy 배열도 같은 마스크로 필터링
            available_indices = available_mask.values
            filtered_embeddings = self.embeddings[available_indices]
        else:
            filtered_df = self.processed_df.reset_index(drop=True)
            filtered_embeddings = self.embeddings
        
        # 3. 하드 필터 적용
        final_df, final_embeddings = self.apply_hard_filters(
            filtered_df, filtered_embeddings, preferences
        )
        
        print(f"📋 필터링: 전체 {len(self.processed_df)}마리 → 입양가능 {len(filtered_df)}마리 → 조건부합 {len(final_df)}마리")
        
        if len(final_df) == 0:
            print("❌ 조건에 맞는 동물이 없습니다.")
            return []
        
        # 4. 성격 유사도 계산
        query_embedding = self.process_user_query(user_query)
        if query_embedding is None:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, final_embeddings).flatten()
        
        # 5. 상위 k개 추출
        available_count = min(top_k, len(similarities))
        top_indices = similarities.argsort()[-available_count:][::-1]
        
        # 6. 결과 정리
        results = []
        for i, idx in enumerate(top_indices):
            # UID를 사용해서 링크 생성
            uid = final_df.iloc[idx].get('uid', '')
            link = f"https://www.pimfyvirus.com/search/01_v/{uid}" if uid else "링크 없음"
            
            animal_info = {
                'rank': i + 1,
                'index': idx,
                'uid': uid,
                'link': link,
                'similarity': similarities[idx],
                'name': final_df.iloc[idx]['addinfo01'],
                'gender': final_df.iloc[idx]['addinfo03'],
                'weight': final_df.iloc[idx]['addinfo07'],
                'age': final_df.iloc[idx].get('addinfo05', '나이미정'),
                'neuter': final_df.iloc[idx].get('addinfo04', '중성화미정'),
                'personality_tags': final_df.iloc[idx]['addinfo08'],
                'personality_desc': final_df.iloc[idx].get('addinfo10', ''),
                'rescue_story': final_df.iloc[idx].get('addinfo09', ''),
                'special_needs': final_df.iloc[idx].get('addinfo16', ''),
                'state': final_df.iloc[idx]['state'],
                'kind': final_df.iloc[idx].get('kind', '임보종류미정'),
            }
            results.append(animal_info)
        
        # 7. 결과 출력
        print(f"\n🎯 추천 결과 (조건 맞춤) (상위 {available_count}개):")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['rank']}. 🐕 {result['name']} (유사도: {result['similarity']:.3f}) 🟢")
            print(f"   📊 기본정보: {result['gender']}, {result['weight']}kg, {result['state']}")
            print(f"   🎭 성격특징: {result['personality_tags']}")
            print(f"   🔗 상세정보: {result['link']}")
            
            # 성격 설명이 있으면 표시
            if result['personality_desc'] and result['personality_desc'].strip():
                desc = result['personality_desc'][:200] + "..." if len(result['personality_desc']) > 200 else result['personality_desc']
                print(f"   💭 성격설명: {desc}")
            
            # 특별 요구사항이 있으면 표시
            if result['special_needs'] and result['special_needs'].strip():
                print(f"   ⚠️  특별요구: {result['special_needs']}")
        
        return results
    
    def save_embeddings(self, output_path="animal_embeddings.pkl"):
        """임베딩 데이터 저장"""
        print(f"\n💾 임베딩 데이터 저장: {output_path}")
        
        data = {
            'embeddings': self.embeddings,
            'dataframe': self.processed_df,
            'model': self.model,
            'embedding_dim': self.embedding_dim,
            'metadata': {
                'total_animals': len(self.processed_df),
                'embedding_shape': self.embeddings.shape,
                'model_used': self.model
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ 저장 완료")
    
    def load_embeddings(self, file_path="animal_embeddings.pkl"):
        """저장된 임베딩 데이터 로드"""
        print(f"📂 임베딩 데이터 로딩: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.processed_df = data['dataframe']
        self.model = data['model']
        self.embedding_dim = data['embedding_dim']
        
        print(f"✅ 로딩 완료: {data['metadata']}")
    
    def get_recommendation_stats(self):
        """추천 시스템 통계 정보"""
        if self.processed_df is None:
            print("❌ 데이터가 로드되지 않았습니다.")
            return
        
        print("\n📊 추천 시스템 통계")
        print("=" * 40)
        
        total_animals = len(self.processed_df)
        available_animals = len(self.processed_df[self.processed_df['state'] == '임보가능'])
        
        print(f"🐕 전체 동물 수: {total_animals:,}마리")
        print(f"✅ 입양 가능: {available_animals:,}마리")
        print(f"📈 입양 가능 비율: {available_animals/total_animals*100:.1f}%")
        
        # 상태별 분포
        state_counts = self.processed_df['state'].value_counts()
        print(f"\n📋 상태별 분포:")
        for state, count in state_counts.items():
            percentage = count/total_animals*100
            print(f"   {state}: {count:,}마리 ({percentage:.1f}%)")

# 사용 예시
if __name__ == "__main__":
    # API 키 설정 (환경변수 또는 직접 입력)
    import os
    api_key = os.getenv('OPENAI_API_KEY') or "your-api-key-here"
    
    # GPT 임베딩 프로세서 생성
    gpt_processor = GPTEmbeddingProcessor(
        api_key=api_key,
        model="text-embedding-3-small"  # 또는 "text-embedding-3-large"
    )
    
    # 전처리된 데이터 로드
    df = pd.read_csv('homeprotection_data.csv')
    
    # 임베딩 생성 (시간 소요)
    embeddings, texts = gpt_processor.process_animal_data(df)
    
    # 임베딩 저장
    gpt_processor.save_embeddings()
    
    # 추천 테스트
    results = gpt_processor.find_similar_animals("활발하고 애교 많은 소형견을 원해요", available_only=True)
    
    print("\n🎯 GPT 임베딩 기반 추천 시스템 구축 완료!")