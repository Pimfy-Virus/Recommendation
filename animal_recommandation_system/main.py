import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 내 모듈 import
from data_preprocessor import AnimalDataProcessorForGPT
from Embedding_processor import GPTEmbeddingProcessor

class AnimalRecommendationMain:
    def __init__(self):
        self.data_file = 'homeprotection_data.csv'
        self.preprocessed_file = 'gpt_preprocessed_data.pkl'
        self.embedding_file = 'animal_embeddings.pkl'
        self.gpt_processor = None
        self.embedding_processor = None
        
        # OpenAI API 키 설정
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("⚠️  OpenAI API 키가 설정되지 않았습니다.")
            print("   환경변수 OPENAI_API_KEY를 설정하거나")
            print("   아래 주석을 해제하여 직접 입력하세요.")
            # self.api_key = "your-api-key-here"  # 직접 입력시 주석 해제
        
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
                # 실제 추천 시스템 호출 - 입양 가능한 동물만
                results = self.embedding_processor.find_similar_animals(
                    query, 
                    top_k=5, 
                    available_only=True  # 입양 가능한 동물만 필터링
                )
                
                print("\n" + "⭐"*20)
                
                # 사용자 입력 대기 (선택사항)
                if i % 3 == 0:  # 3개마다 일시정지
                    user_input = input("\n⏸️  계속하려면 Enter, 종료하려면 'q': ")
                    if user_input.lower() == 'q':
                        print("테스트를 종료합니다.")
                        break
            
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