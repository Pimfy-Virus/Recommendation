import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import openai
from dotenv import load_dotenv

# .env 파일을 명시적으로 로드
env_path = '/Users/sxxwings/git/pimfy/Recommendation/.env'
print(f"🔍 .env 파일 로딩: {env_path}")
load_dotenv(env_path)

# 프로젝트 내 모듈 import
from data_preprocessor import AnimalDataProcessorForGPT
from embedding_processor import GPTEmbeddingProcessor

class AnimalRecommendationMain:
    def __init__(self):
        # 현재 스크립트가 있는 디렉토리를 기준으로 파일 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(script_dir, 'homeprotection_data.csv')
        self.preprocessed_file = os.path.join(script_dir, 'gpt_preprocessed_data.pkl')
        self.embedding_file = os.path.join(script_dir, 'animal_embeddings.pkl')
        self.gpt_processor = None
        self.embedding_processor = None
        
        # OpenAI API 키 설정
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        # API 키 유효성 검사
        if not self.api_key or self.api_key == 'your-openai-api-key-here':
            print("⚠️  OpenAI API 키가 설정되지 않았습니다.")
            print("   환경변수 OPENAI_API_KEY를 설정하거나")
            print("   .env 파일에서 실제 API 키로 수정하세요.")
            print("   현재 API 키:", self.api_key[:20] + "..." if self.api_key else "None")
            # self.api_key = "your-api-key-here"  # 직접 입력시 주석 해제
        else:
            # OpenAI 클라이언트 설정
            openai.api_key = self.api_key
            print(f"✅ OpenAI API 키 설정 완료: {self.api_key[:8]}...")
        
        # 테스트용 사용자 쿼리들
        self.test_queries = [
            "활발하고 애교 많은 소형견을 원해요",
            "조용하고 얌전한 중형 강아지가 좋겠어요", 
            "사람을 좋아하고 똑똑한 아이를 찾고 있어요",
            "에너지가 넘치고 장난기 많은 강아지",
            "소심하지만 착한 성격의 동물",
            "산책을 좋아하고 활동적인 개",
            "아파트에서 키우기 좋은 조용한 아이",
            "아이들과 잘 지낼 수 있는 친화적인 강아지",
            "첫 반려동물로 키우기 쉬운 순한 아이",
            "나이가 좀 있어도 괜찮으니까 차분한 성격의 개"
        ]
    
    def check_files(self):
        """필요한 파일들 존재 여부 확인"""
        print("📁 파일 확인 중...")
        
        files_status = {
            '원본 데이터': (self.data_file, os.path.exists(self.data_file)),
            '전처리된 데이터': (self.preprocessed_file, os.path.exists(self.preprocessed_file)),
            '임베딩 데이터': (self.embedding_file, os.path.exists(self.embedding_file))
        }
        
        for name, (file_path, exists) in files_status.items():
            status = "✅ 존재" if exists else "❌ 없음"
            print(f"   {name}: {status} ({file_path})")
        
        return files_status
    
    def setup_processors(self):
        """프로세서 객체 초기화"""
        print("\n🔧 프로세서 초기화 중...")
        
        try:
            # GPT 전처리기
            self.gpt_processor = AnimalDataProcessorForGPT()
            
            # GPT 임베딩 프로세서
            if self.api_key:
                self.embedding_processor = GPTEmbeddingProcessor(
                    api_key=self.api_key,
                    model="text-embedding-3-large"  # 파일에서 large 모델 사용
                )
                print("   ✅ GPT 임베딩 프로세서 초기화 완료")
            else:
                print("   ❌ API 키가 없어 GPT 임베딩 프로세서 초기화 실패")
                return False
                
        except Exception as e:
            print(f"   ❌ 프로세서 초기화 실패: {e}")
            return False
        
        return True
    
    def run_preprocessing(self):
        """데이터 전처리 실행"""
        print("\n" + "="*60)
        print("🔄 데이터 전처리 단계")
        print("="*60)
        
        if not os.path.exists(self.data_file):
            print(f"❌ 원본 데이터 파일이 없습니다: {self.data_file}")
            return False
        
        try:
            # GPT용 전처리 실행
            print("📝 GPT 임베딩용 전처리 시작...")
            df, descriptions = self.gpt_processor.process_for_gpt_embedding(self.data_file)
            self.gpt_processor.save_processed_data(self.preprocessed_file)
            
            print("   ✅ 전처리 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 전처리 중 오류 발생: {e}")
            return False
    
    def run_embedding_generation(self):
        """GPT 임베딩 생성"""
        print("\n" + "="*60)
        print("🤖 GPT 임베딩 생성 단계")
        print("="*60)
        
        if not os.path.exists(self.preprocessed_file):
            print(f"❌ 전처리된 데이터가 없습니다: {self.preprocessed_file}")
            return False
        
        try:
            # 전처리된 데이터 로드
            print("📂 전처리된 데이터 로딩...")
            with open(self.preprocessed_file, 'rb') as f:
                data = pickle.load(f)
            df = data['dataframe']
            
            # GPT 임베딩 생성
            print("🚀 GPT 임베딩 생성 시작...")
            print("   ⏱️  예상 소요 시간: 5-10분 (데이터 크기에 따라)")
            
            embeddings, texts = self.embedding_processor.process_animal_data(df)
            self.embedding_processor.save_embeddings(self.embedding_file)
            
            print("   ✅ 임베딩 생성 완료")
            
            return True
            
        except Exception as e:
            print(f"❌ 임베딩 생성 중 오류 발생: {e}")
            return False
    
    def load_system(self):
        """기존 시스템 로드"""
        print("\n" + "="*60)
        print("📂 기존 시스템 로드")
        print("="*60)
        
        try:
            if os.path.exists(self.embedding_file):
                self.embedding_processor.load_embeddings(self.embedding_file)
                print("   ✅ 임베딩 데이터 로드 완료")
                return True
            else:
                print(f"   ❌ 임베딩 파일이 없습니다: {self.embedding_file}")
                return False
                
        except Exception as e:
            print(f"❌ 시스템 로드 중 오류 발생: {e}")
            return False
    
    def run_recommendations(self):
        """추천 시스템 테스트 실행"""
        print("\n" + "="*60)
        print("🎯 추천 시스템 테스트")
        print("="*60)
        
        print("🧪 테스트 모드로 실행합니다.")
        print(f"📝 준비된 테스트 쿼리: {len(self.test_queries)}개\n")
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n【테스트 {i}/{len(self.test_queries)}】")
            print(f"🔍 사용자 쿼리: '{query}'")
            print("-" * 50)
            
            try:
                # 1단계: 코사인 유사도 기반 초기 후보 선정
                initial_candidates = self.embedding_processor.find_similar_animals(
                    query, 
                    top_k=10,  # 더 많은 후보를 가져와서 GPT가 선택할 수 있도록
                    available_only=True
                )
                
                if not initial_candidates:
                    print("❌ 추천할 동물이 없습니다.")
                    continue
                
                print(f"\n🔍 1단계: 유사도 기반 {len(initial_candidates)}마리 후보 선정 완료")
                
                # 2단계: GPT에게 재추천 요청
                print("🤖 2단계: GPT 재추천 진행 중...")
                gpt_response = self.get_llm_recommendations(query, initial_candidates)
                
                if gpt_response:
                    print("\n" + "🎯" * 20)
                    print("【GPT 최종 추천 결과】")
                    print("=" * 50)
                    print(gpt_response)
                    
                    # GPT 응답 파싱 (선택사항)
                    parsed_recommendations = self.parse_gpt_recommendations(gpt_response)
                    
                    if parsed_recommendations:
                        print("\n📋 파싱된 추천 결과:")
                        for j, rec in enumerate(parsed_recommendations, 1):
                            if 'name' in rec and 'reason' in rec:
                                print(f"{j}. {rec['name']}")
                                print(f"   💡 {rec['reason']}")
                        
                        # 3단계: 사용자 만족도 확인 및 구글 시트 저장
                        print("\n" + "📝" * 20)
                        try:
                            satisfaction_input = input("이 추천 결과에 만족하시나요? (1-5점, Enter=저장안함): ").strip()
                            
                            satisfaction = None
                            if satisfaction_input and satisfaction_input.isdigit():
                                satisfaction = int(satisfaction_input)
                                if 1 <= satisfaction <= 5:
                                    print(f"✅ 만족도 {satisfaction}점으로 기록됩니다.")
                                    
                                    # 구글 시트(CSV)에 저장
                                    self.save_to_google_sheets(query, parsed_recommendations, satisfaction)
                                else:
                                    print("⚠️ 1-5점 범위를 벗어나서 저장하지 않습니다.")
                            else:
                                print("📋 만족도 없이 진행합니다.")
                                
                        except KeyboardInterrupt:
                            print("\n⏸️ 입력을 건너뜁니다.")
                        except Exception as e:
                            print(f"❌ 만족도 처리 중 오류: {e}")
                    
                else:
                    print("❌ GPT 재추천 실패. 초기 유사도 결과를 표시합니다.")
                    print("\n🔍 유사도 기반 추천 결과:")
                    for j, candidate in enumerate(initial_candidates[:5], 1):
                        print(f"{j}. {candidate['name']} (유사도: {candidate['similarity']:.3f})")
                
                print("\n" + "⭐"*20)
                
                # 사용자 입력 대기 (선택사항)
                if i % 1 == 0:  # 매번 일시정지해서 Rate limit 방지
                    user_input = input("\n⏸️  계속하려면 Enter, 종료하려면 'q': ")
                    if user_input.lower() == 'q':
                        print("테스트를 종료합니다.")
                        break
                    
                # Rate limit 방지를 위한 추가 대기
                print("⏱️ API 호출 제한 방지를 위해 3초 대기 중...")
                time.sleep(3)
            
            except Exception as e:
                print(f"❌ 추천 처리 중 오류: {e}")
                continue
    
    def show_system_stats(self):
        """시스템 통계 정보 표시"""
        print("\n" + "="*60)
        print("📊 시스템 통계")
        print("="*60)
        
        try:
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file)
                print(f"📈 총 동물 수: {len(df):,}마리")
                
                # 상태별 통계
                if 'state' in df.columns:
                    state_counts = df['state'].value_counts()
                    print(f"📋 상태별 분포:")
                    for state, count in state_counts.items():
                        print(f"   {state}: {count:,}마리")
                
                # 성별 통계
                if 'addinfo03' in df.columns:
                    gender_counts = df['addinfo03'].value_counts()
                    print(f"🚻 성별 분포:")
                    for gender, count in gender_counts.items():
                        print(f"   {gender}: {count:,}마리")
            
            # 파일 크기 정보
            files = [self.data_file, self.preprocessed_file, self.embedding_file]
            print(f"\n💾 파일 크기:")
            for file_path in files:
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / 1024 / 1024
                    print(f"   {file_path}: {size_mb:.1f} MB")
                    
        except Exception as e:
            print(f"❌ 통계 조회 중 오류: {e}")
    
    def run_full_pipeline(self, skip_existing=True):
        """전체 파이프라인 실행"""
        print("🚀 동물 추천 시스템 전체 파이프라인 시작")
        print("=" * 80)
        print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 파일 확인
        files_status = self.check_files()
        
        # 2. 프로세서 초기화
        if not self.setup_processors():
            print("❌ 프로세서 초기화 실패. 프로그램을 종료합니다.")
            return False
        
        # 3. 전처리 (필요시)
        if not files_status['전처리된 데이터'][1] or not skip_existing:
            if not self.run_preprocessing():
                print("❌ 전처리 실패. 프로그램을 종료합니다.")
                return False
        else:
            print("✅ 전처리된 데이터 존재 - 스킵")
        
        # 4. 임베딩 생성 (필요시)
        if not files_status['임베딩 데이터'][1] or not skip_existing:
            if not self.run_embedding_generation():
                print("❌ 임베딩 생성 실패. 프로그램을 종료합니다.")
                return False
        else:
            print("✅ 임베딩 데이터 존재 - 스킵")
        
        # 5. 시스템 로드
        if not self.load_system():
            print("❌ 시스템 로드 실패. 프로그램을 종료합니다.")
            return False
        
        # 6. 시스템 통계
        self.show_system_stats()
        
        # 7. 추천 테스트
        self.run_recommendations()
        
        print(f"\n⏰ 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 전체 파이프라인 완료!")
        
        return True
    
    def build_recommendation_prompt(self, query, candidates):
        """GPT 추천을 위한 프롬프트 생성"""
        prompt = f"""사용자 쿼리:
"{query}"

아래는 추천 후보 동물 리스트입니다:"""
        
        for i, animal in enumerate(candidates, 1):
            prompt += f"""
{i}. 이름: {animal['name']}
   설명: {animal['description']}
   유사도 점수: {animal['similarity']:.3f}
"""
        
        prompt += """

위 정보와 사용자 쿼리를 바탕으로
1) 가장 적합하다고 판단되는 5마리를 순서대로 추천해 주세요.
2) 각 추천에 대해 이유를 2-3문장으로 작성해 주세요.
3) 출력 형식은 아래와 같이 해주세요:

추천 리스트:
1. 이름: (이름)
   이유: (이유)
2. 이름: (이름)
   이유: (이유)
3. ...
"""
        return prompt
    
    def get_llm_recommendations(self, query, initial_candidates):
        """GPT에게 후보군을 전달해서 재추천받기"""
        try:
            import openai
            
            # 후보 동물들을 GPT용 형식으로 변환
            candidates_for_gpt = []
            for candidate in initial_candidates:
                # 임베딩 텍스트나 상세 설명 가져오기
                description = self.get_animal_description(candidate)
                
                candidates_for_gpt.append({
                    'name': candidate['name'],
                    'description': description,
                    'similarity': candidate['similarity']
                })
            
            # GPT 프롬프트 생성
            prompt = self.build_recommendation_prompt(query, candidates_for_gpt)
            
            print("🤖 GPT에게 재추천 요청 중...")
            
            # GPT API 호출
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # 비용 효율적인 모델 사용
                messages=[
                    {"role": "system", "content": "당신은 동물 입양 전문가입니다. 사용자의 요구사항에 가장 적합한 동물을 추천해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 일관성 있는 추천을 위해 낮은 temperature
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ GPT 재추천 중 오류: {e}")
            return None
    
    def get_animal_description(self, candidate):
        """동물의 상세 설명 생성"""
        description_parts = []
        
        # 기본 정보
        description_parts.append(f"성별: {candidate['gender']}, 몸무게: {candidate['weight']}kg")
        
        # 성격 특징
        if candidate.get('personality_tags'):
            description_parts.append(f"성격: {candidate['personality_tags']}")
        
        # 성격 상세 설명
        if candidate.get('personality_desc'):
            desc = candidate['personality_desc'][:200] + "..." if len(candidate['personality_desc']) > 200 else candidate['personality_desc']
            description_parts.append(f"상세설명: {desc}")
        
        # 특별 요구사항
        if candidate.get('special_needs'):
            description_parts.append(f"특별요구: {candidate['special_needs']}")
        
        # 구조 배경
        if candidate.get('rescue_story'):
            story = candidate['rescue_story'][:150] + "..." if len(candidate['rescue_story']) > 150 else candidate['rescue_story']
            description_parts.append(f"구조배경: {story}")
        
        return " | ".join(description_parts)
    
    def parse_gpt_recommendations(self, gpt_response):
        """GPT 응답을 파싱해서 구조화된 데이터로 변환"""
        recommendations = []
        
        try:
            lines = gpt_response.split('\n')
            current_rec = {}
            
            for line in lines:
                line = line.strip()
                
                # 추천 번호와 이름 파싱
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    # 이전 추천 저장
                    if current_rec:
                        recommendations.append(current_rec)
                    
                    # 새 추천 시작
                    current_rec = {}
                    if '이름:' in line:
                        name_part = line.split('이름:')[-1].strip()
                        current_rec['name'] = name_part
                
                # 이유 파싱
                elif line.startswith('이유:') and current_rec:
                    reason = line.replace('이유:', '').strip()
                    current_rec['reason'] = reason
            
            # 마지막 추천 저장
            if current_rec:
                recommendations.append(current_rec)
                
        except Exception as e:
            print(f"❌ GPT 응답 파싱 중 오류: {e}")
        
        return recommendations
    
    def save_to_google_sheets(self, query, recommendations, user_satisfaction=None):
        """추천 결과를 구글 시트에 저장"""
        try:
            # 구글 시트 연동은 별도 라이브러리 필요
            # 여기서는 CSV 파일로 대체 구현
            import csv
            from datetime import datetime
            
            # CSV 파일 경로
            results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recommendation_results.csv')
            
            # 데이터 준비
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # CSV 파일이 없으면 헤더 생성
            file_exists = os.path.exists(results_file)
            
            with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'user_query', 'rank', 'animal_name', 'reason', 'satisfaction']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 헤더 작성 (파일이 새로 생성된 경우)
                if not file_exists:
                    writer.writeheader()
                
                # 추천 결과 저장
                for i, rec in enumerate(recommendations, 1):
                    writer.writerow({
                        'timestamp': timestamp,
                        'user_query': query,
                        'rank': i,
                        'animal_name': rec.get('name', '이름없음'),
                        'reason': rec.get('reason', '이유없음'),
                        'satisfaction': user_satisfaction
                    })
            
            print(f"💾 추천 결과가 저장되었습니다: {results_file}")
            return True
            
        except Exception as e:
            print(f"❌ 결과 저장 중 오류: {e}")
            return False
    
    def setup_google_sheets_integration(self):
        """구글 시트 연동 설정 (향후 확장용)"""
        """
        실제 구글 시트 연동을 위해서는 다음이 필요합니다:
        1. pip install gspread google-auth
        2. Google Cloud Console에서 서비스 계정 생성
        3. 서비스 계정 키 파일 다운로드
        4. 구글 시트 생성 및 서비스 계정에 편집 권한 부여
        
        예시 코드:
        import gspread
        from google.oauth2.service_account import Credentials
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file("path/to/credentials.json", scopes=scope)
        client = gspread.authorize(creds)
        
        sheet = client.open("동물 추천 결과").sheet1
        sheet.append_row([timestamp, query, animal_name, reason, satisfaction])
        """
        print("📊 구글 시트 연동 기능은 현재 CSV 파일로 대체 구현되었습니다.")
        print("   실제 구글 시트 연동을 원하시면 위 주석의 가이드를 참고하세요.")
        
    def test_single_recommendation(self, query):
        """단일 쿼리로 추천 시스템 테스트"""
        print("\n" + "="*60)
        print("🎯 단일 추천 테스트")
        print("="*60)
        print(f"🔍 사용자 쿼리: '{query}'")
        print("-" * 50)
        
        try:
            # 1단계: 코사인 유사도 기반 초기 후보 선정
            print("1️⃣ 유사도 기반 후보 선정 중...")
            initial_candidates = self.embedding_processor.find_similar_animals(
                query, 
                top_k=10,
                available_only=True
            )
            
            if not initial_candidates:
                print("❌ 추천할 동물이 없습니다.")
                return False
            
            print(f"✅ {len(initial_candidates)}마리 후보 선정 완료")
            
            # 2단계: GPT에게 재추천 요청
            print("2️⃣ GPT 재추천 진행 중...")
            gpt_response = self.get_llm_recommendations(query, initial_candidates)
            
            if gpt_response:
                print("\n" + "🎯" * 20)
                print("【GPT 최종 추천 결과】")
                print("=" * 50)
                print(gpt_response)
                
                # GPT 응답 파싱
                parsed_recommendations = self.parse_gpt_recommendations(gpt_response)
                
                if parsed_recommendations:
                    print("\n📋 파싱된 추천 결과:")
                    for j, rec in enumerate(parsed_recommendations, 1):
                        if 'name' in rec and 'reason' in rec:
                            print(f"{j}. {rec['name']}")
                            print(f"   💡 {rec['reason']}")
                else:
                    print("⚠️ GPT 응답 파싱에 실패했습니다.")
                
                return True
            else:
                print("❌ GPT 재추천 실패")
                return False
                
        except Exception as e:
            print(f"❌ 추천 처리 중 오류: {e}")
            return False

def main():
    """메인 실행 함수"""
    print("🐕 동물 임시보호 추천 시스템 v1.0")
    print("=" * 80)
    
    # 추천 시스템 객체 생성
    recommender = AnimalRecommendationMain()
    
    try:
        # 전체 파이프라인 실행
        recommender.run_full_pipeline(skip_existing=True)
            
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()