"""
임시보호 동물 추천 시스템 메인 실행 파일
데이터 전처리, 필터링, 추천 기능을 통합 제공
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

from data_preprocessor import DataPreprocessor
from animal_filter import AnimalFilter


class AnimalRecommendationSystem:
    """임시보호 동물 추천 시스템 메인 클래스"""
    
    def __init__(self, csv_path: str = None):
        self.preprocessor = DataPreprocessor()
        self.filter = AnimalFilter()
        self.processed_data = None
        self.metadata = None
        
        if csv_path and os.path.exists(csv_path):
            self.load_data(csv_path)
    
    def load_data(self, csv_path: str):
        """데이터 로드 및 전처리"""
        print("=" * 50)
        print("임시보호 동물 데이터 로딩 중...")
        print("=" * 50)
        
        try:
            # 데이터 전처리
            self.processed_data = self.preprocessor.load_and_process(csv_path)
            self.metadata = self.preprocessor.get_metadata()
            
            # 필터에 데이터 설정
            self.filter.set_animals(self.processed_data)
            
            print("\n✅ 데이터 로딩 완료!")
            self.show_data_summary()
            
        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            sys.exit(1)
    
    def show_data_summary(self):
        """데이터 요약 정보 출력"""
        stats = self.preprocessor.get_statistics()
        
        print(f"\n📊 데이터 요약:")
        print(f"  • 전체 동물 수: {stats['total']:,}마리")
        print(f"  • 임보 가능: {stats['available']:,}마리")
        print(f"  • 평균 나이: {stats['average_age']}세")
        print(f"  • 평균 몸무게: {stats['average_weight']}kg")
        
        print(f"\n🏷️ 성별 분포:")
        for gender, count in stats['gender_distribution'].items():
            print(f"  • {gender}: {count:,}마리")
        
        print(f"\n🏠 임보 종류 분포:")
        for care_type, count in list(stats['care_type_distribution'].items())[:5]:
            print(f"  • {care_type}: {count:,}마리")
        
        print(f"\n📍 주요 구조 지역:")
        for region, count in list(stats['region_distribution'].items())[:5]:
            print(f"  • {region}: {count:,}마리")
    
    def interactive_filtering(self):
        """대화형 필터링 인터페이스"""
        print("\n" + "=" * 50)
        print("🔍 동물 필터링 시작")
        print("=" * 50)
        
        filter_criteria = {}
        
        # 지역 선택
        print(f"\n📍 구조 지역 선택 (전체: {len(self.metadata['regions'])}개)")
        print("주요 지역:", ', '.join(list(self.metadata['regions'])[:10]))
        region_input = input("원하는 지역을 입력하세요 (없으면 Enter): ").strip()
        if region_input:
            filter_criteria['region'] = [region_input]
        
        # 성별 선택
        print(f"\n⚥ 성별 선택")
        print("옵션: male, female")
        gender_input = input("원하는 성별을 입력하세요 (없으면 Enter): ").strip()
        if gender_input:
            filter_criteria['gender'] = [gender_input]
        
        # 나이 범위 선택
        print(f"\n🎂 나이 범위 선택")
        min_age = input("최소 나이 (없으면 Enter): ").strip()
        max_age = input("최대 나이 (없으면 Enter): ").strip()
        if min_age or max_age:
            age_range = {}
            if min_age.isdigit():
                age_range['min'] = int(min_age)
            if max_age.isdigit():
                age_range['max'] = int(max_age)
            if age_range:
                filter_criteria['age_range'] = age_range
        
        # 몸무게 범위 선택
        print(f"\n⚖️ 몸무게 범위 선택 (kg)")
        min_weight = input("최소 몸무게 (없으면 Enter): ").strip()
        max_weight = input("최대 몸무게 (없으면 Enter): ").strip()
        if min_weight or max_weight:
            weight_range = {}
            try:
                if min_weight:
                    weight_range['min'] = float(min_weight)
                if max_weight:
                    weight_range['max'] = float(max_weight)
                if weight_range:
                    filter_criteria['weight_range'] = weight_range
            except ValueError:
                print("⚠️ 올바른 숫자를 입력해주세요.")
        
        # 중성화 여부
        print(f"\n✂️ 중성화 여부")
        print("옵션: yes (중성화 완료), no (중성화 안함)")
        neutered_input = input("중성화 여부 (없으면 Enter): ").strip().lower()
        if neutered_input in ['yes', 'y']:
            filter_criteria['neutered'] = True
        elif neutered_input in ['no', 'n']:
            filter_criteria['neutered'] = False
        
        # 성격 해시태그
        print(f"\n🏷️ 원하는 성격 특성")
        print("예시 해시태그:", ', '.join(self.metadata['all_hashtags'][:10]))
        hashtag_input = input("원하는 성격을 입력하세요 (쉼표로 구분, 없으면 Enter): ").strip()
        if hashtag_input:
            hashtags = [tag.strip() for tag in hashtag_input.split(',')]
            filter_criteria['hashtags'] = hashtags
        
        return filter_criteria
    
    def apply_hard_filtering(self, filter_criteria: Dict):
        """하드 필터링 적용"""
        print(f"\n🔍 필터링 조건 적용 중...")
        
        results = self.filter.apply_filters(filter_criteria)
        
        print(f"✅ 필터링 완료: {len(results)}마리 발견")
        
        if len(results) > 0:
            self.show_filter_results(results)
            return results
        else:
            print("😿 조건에 맞는 동물이 없습니다. 조건을 완화해보세요.")
            return None
    
    def apply_smart_recommendation(self):
        """스마트 추천 시스템"""
        print("\n" + "=" * 50)
        print("🤖 AI 기반 스마트 추천")
        print("=" * 50)
        
        preferences = self.get_user_preferences()
        
        print(f"\n🧠 추천 점수 계산 중...")
        recommendations = self.filter.apply_soft_filtering(preferences, threshold=0.3)
        
        print(f"✅ 추천 완료: {len(recommendations)}마리")
        
        if len(recommendations) > 0:
            self.show_recommendations(recommendations)
            return recommendations
        else:
            print("😿 추천할 수 있는 동물이 없습니다.")
            return None
    
    def get_user_preferences(self) -> Dict:
        """사용자 선호도 수집"""
        preferences = {}
        
        # 나이 선호도
        print(f"\n🎂 선호하는 나이대")
        pref_min_age = input("선호 최소 나이: ").strip()
        pref_max_age = input("선호 최대 나이: ").strip()
        accept_min_age = input("허용 최소 나이 (선호보다 넓게): ").strip()
        accept_max_age = input("허용 최대 나이 (선호보다 넓게): ").strip()
        
        if pref_min_age.isdigit() and pref_max_age.isdigit():
            age_pref = {
                'preferred': {'min': int(pref_min_age), 'max': int(pref_max_age)}
            }
            if accept_min_age.isdigit() and accept_max_age.isdigit():
                age_pref['acceptable'] = {'min': int(accept_min_age), 'max': int(accept_max_age)}
            preferences['age_preference'] = age_pref
        
        # 크기 선호도
        print(f"\n⚖️ 선호하는 크기 (kg)")
        pref_min_weight = input("선호 최소 몸무게: ").strip()
        pref_max_weight = input("선호 최대 몸무게: ").strip()
        
        if pref_min_weight and pref_max_weight:
            try:
                size_pref = {
                    'preferred': {'min': float(pref_min_weight), 'max': float(pref_max_weight)},
                    'acceptable': {'min': 0, 'max': 100}  # 기본 허용 범위
                }
                preferences['size_preference'] = size_pref
            except ValueError:
                pass
        
        # 성격 선호도
        print(f"\n🏷️ 원하는 성격 특성")
        print("예시:", ', '.join(self.metadata['all_hashtags'][:15]))
        personality_input = input("원하는 성격들을 입력하세요 (쉼표로 구분): ").strip()
        if personality_input:
            personalities = [p.strip() for p in personality_input.split(',')]
            preferences['personality_traits'] = personalities
        
        # 행동 특성 선호도
        print(f"\n🐕 행동 특성 선호도 (1-5 점수)")
        behavior_prefs = {}
        
        behavior_questions = {
            'affection': '애정 표현 (1: 별로, 5: 매우 좋아함)',
            'human_friendly': '사람 친화성 (1: 낯가림, 5: 매우 친화적)',
            'barking': '짖음 정도 (1: 거의 안짖음, 5: 자주 짖음)'
        }
        
        for trait, question in behavior_questions.items():
            score = input(f"{question}: ").strip()
            if score.isdigit() and 1 <= int(score) <= 5:
                behavior_prefs[trait] = {
                    'ideal': int(score),
                    'acceptable': [max(1, int(score)-1), int(score), min(5, int(score)+1)]
                }
        
        if behavior_prefs:
            preferences['behavior_preferences'] = behavior_prefs
        
        # 가중치 설정
        preferences['weights'] = {
            'age': 1.5,
            'size': 1.2,
            'personality': 1.8,
            'behavior': 1.3
        }
        
        return preferences
    
    def show_filter_results(self, results):
        """필터 결과 표시"""
        print(f"\n📋 필터링 결과 (상위 10마리)")
        print("-" * 80)
        
        for i, (idx, animal) in enumerate(results.head(10).iterrows()):
            print(f"{i+1:2d}. {animal['name']} ({animal['gender']}, {animal['age']}세, {animal['weight']}kg)")
            print(f"    📍 {animal['rescue_location']} | 🏠 {animal['care_type']}")
            print(f"    🏷️ {', '.join(animal['hashtags'][:3])}")
            print(f"    🔗 {animal['detail_link']}")
            print()
        
        # 통계 표시
        stats = self.filter.get_result_stats()
        print(f"📊 결과 통계:")
        print(f"  성별: {stats.get('gender_distribution', {})}")
        print(f"  나이: {stats.get('age_distribution', {})}")
    
    def show_recommendations(self, recommendations):
        """추천 결과 표시"""
        print(f"\n🎯 추천 결과 (상위 10마리)")
        print("-" * 80)
        
        for i, (idx, animal) in enumerate(recommendations.head(10).iterrows()):
            match_score = animal.get('match_score', 0)
            print(f"{i+1:2d}. {animal['name']} (매칭도: {match_score:.1%})")
            print(f"    👤 {animal['gender']}, {animal['age']}세, {animal['weight']}kg")
            print(f"    📍 {animal['rescue_location']} | 🏠 {animal['care_type']}")
            print(f"    🏷️ {', '.join(animal['hashtags'][:3])}")
            print(f"    🔗 {animal['detail_link']}")
            print()
    
    def save_results(self, results, filename_prefix="results"):
        """결과 저장"""
        if results is not None and len(results) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.csv"
            self.filter.export_results(filename)
            return filename
        return None
    
    def run_interactive_mode(self):
        """대화형 모드 실행"""
        while True:
            print("\n" + "=" * 50)
            print("🐕 임시보호 동물 추천 시스템")
            print("=" * 50)
            print("1. 조건별 필터링")
            print("2. AI 스마트 추천")
            print("3. 데이터 통계 보기")
            print("4. 종료")
            
            choice = input("\n선택하세요 (1-4): ").strip()
            
            if choice == '1':
                filter_criteria = self.interactive_filtering()
                results = self.apply_hard_filtering(filter_criteria)
                
                if results is not None:
                    save_choice = input("\n결과를 저장하시겠습니까? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        saved_file = self.save_results(results, "filter_results")
                        if saved_file:
                            print(f"✅ 결과가 {saved_file}에 저장되었습니다.")
            
            elif choice == '2':
                recommendations = self.apply_smart_recommendation()
                
                if recommendations is not None:
                    save_choice = input("\n추천 결과를 저장하시겠습니까? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        saved_file = self.save_results(recommendations, "recommendations")
                        if saved_file:
                            print(f"✅ 추천 결과가 {saved_file}에 저장되었습니다.")
            
            elif choice == '3':
                self.show_data_summary()
                
                # 추가 상세 통계
                print(f"\n📈 상세 통계:")
                print(f"  • 총 해시태그 종류: {len(self.metadata['all_hashtags'])}개")
                print(f"  • 구조 지역 수: {len(self.metadata['regions'])}개")
                print(f"  • 적합한 가정 유형: {len(self.metadata['suitable_home_types'])}개")
                
                # 행동 특성 분포
                behavior_stats = self.get_behavior_statistics()
                print(f"\n🐕 행동 특성 평균 점수:")
                for trait, avg_score in behavior_stats.items():
                    if avg_score is not None:
                        print(f"  • {trait}: {avg_score:.1f}/5.0")
            
            elif choice == '4':
                print("\n👋 시스템을 종료합니다. 좋은 하루 되세요!")
                break
            
            else:
                print("❌ 올바른 선택이 아닙니다. 1-4 중에서 선택해주세요.")
            
            input("\n계속하려면 Enter를 눌러주세요...")
    
    def get_behavior_statistics(self) -> Dict:
        """행동 특성 통계 계산"""
        behavior_traits = [
            'toilet_training', 'walking_needs', 'barking', 'separation_anxiety',
            'shedding', 'affection', 'human_friendly', 'dog_friendly', 'solo_living', 'cat_friendly'
        ]
        
        stats = {}
        for trait in behavior_traits:
            # 각 행동 특성의 평균 점수 계산
            trait_values = []
            for _, animal in self.processed_data.iterrows():
                behavior_data = animal.get('behavior_traits', {})
                if isinstance(behavior_data, dict) and trait in behavior_data:
                    value = behavior_data[trait]
                    if value is not None:
                        trait_values.append(value)
            
            if trait_values:
                stats[trait] = sum(trait_values) / len(trait_values)
            else:
                stats[trait] = None
        
        return stats
    
    def batch_processing_mode(self, user_profiles_file: str):
        """배치 처리 모드 - 여러 사용자 프로필을 한번에 처리"""
        print(f"\n📁 배치 처리 모드 시작")
        print(f"프로필 파일: {user_profiles_file}")
        
        try:
            with open(user_profiles_file, 'r', encoding='utf-8') as f:
                user_profiles = json.load(f)
            
            results_summary = []
            
            for i, profile in enumerate(user_profiles):
                user_id = profile.get('user_id', f'user_{i+1}')
                print(f"\n처리 중: {user_id}")
                
                # 하드 필터링
                if 'hard_filters' in profile:
                    hard_results = self.filter.apply_filters(profile['hard_filters'])
                    print(f"  하드 필터링: {len(hard_results)}마리")
                else:
                    hard_results = self.processed_data[self.processed_data['status'] == '임보가능']
                
                # 소프트 필터링 (추천)
                if 'preferences' in profile:
                    self.filter.set_animals(hard_results)
                    recommendations = self.filter.apply_soft_filtering(profile['preferences'])
                    print(f"  추천 결과: {len(recommendations)}마리")
                    
                    # 상위 5개 추천 저장
                    top_recommendations = recommendations.head(5)
                    
                    results_summary.append({
                        'user_id': user_id,
                        'hard_filter_count': len(hard_results),
                        'recommendation_count': len(recommendations),
                        'top_recommendations': [
                            {
                                'name': rec['name'],
                                'id': rec['id'],
                                'match_score': rec.get('match_score', 0),
                                'detail_link': rec['detail_link']
                            }
                            for _, rec in top_recommendations.iterrows()
                        ]
                    })
                    
                    # 개별 사용자 결과 저장
                    if len(recommendations) > 0:
                        filename = f"recommendations_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        self.filter.export_results(filename)
            
            # 전체 요약 저장
            summary_filename = f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 배치 처리 완료!")
            print(f"요약 파일: {summary_filename}")
            
        except Exception as e:
            print(f"❌ 배치 처리 실패: {e}")


def create_sample_user_profiles():
    """샘플 사용자 프로필 생성"""
    sample_profiles = [
        {
            "user_id": "user_001",
            "hard_filters": {
                "age_range": {"min": 1, "max": 5},
                "weight_range": {"min": 3, "max": 15},
                "gender": ["male", "female"]
            },
            "preferences": {
                "age_preference": {
                    "preferred": {"min": 2, "max": 4},
                    "acceptable": {"min": 1, "max": 6}
                },
                "size_preference": {
                    "preferred": {"min": 5, "max": 12},
                    "acceptable": {"min": 3, "max": 20}
                },
                "personality_traits": ["애교쟁이", "사람좋아", "순둥이"],
                "behavior_preferences": {
                    "affection": {"ideal": 4, "acceptable": [3, 4, 5]},
                    "human_friendly": {"ideal": 5, "acceptable": [4, 5]},
                    "barking": {"ideal": 2, "acceptable": [1, 2, 3]}
                },
                "weights": {
                    "age": 1.5,
                    "size": 1.2,
                    "personality": 1.8,
                    "behavior": 1.3
                }
            }
        },
        {
            "user_id": "user_002",
            "hard_filters": {
                "neutered": True,
                "care_type": ["일반임보", "단기임보"]
            },
            "preferences": {
                "age_preference": {
                    "preferred": {"min": 3, "max": 8},
                    "acceptable": {"min": 1, "max": 10}
                },
                "personality_traits": ["조용조용", "똑똑이"],
                "behavior_preferences": {
                    "barking": {"ideal": 1, "acceptable": [1, 2]},
                    "separation_anxiety": {"ideal": 1, "acceptable": [1, 2]}
                },
                "weights": {
                    "age": 1.0,
                    "personality": 2.0,
                    "behavior": 1.5
                }
            }
        }
    ]
    
    with open('sample_user_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(sample_profiles, f, ensure_ascii=False, indent=2)
    
    print("✅ 샘플 사용자 프로필이 'sample_user_profiles.json'에 생성되었습니다.")


def main():
    """메인 함수"""
    print("🐕 임시보호 동물 추천 시스템 v1.0")
    print("=" * 50)
    
    # CSV 파일 경로 확인
    csv_path = 'pimfyvirus_dog_data.csv'
    if not os.path.exists(csv_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")
        print("데이터 파일을 같은 폴더에 위치시켜주세요.")
        return
    
    # 시스템 초기화
    system = AnimalRecommendationSystem(csv_path)
    
    # 실행 모드 선택
    print("\n실행 모드를 선택하세요:")
    print("1. 대화형 모드 (권장)")
    print("2. 배치 처리 모드")
    print("3. 샘플 프로필 생성")
    
    mode = input("\n선택 (1-3): ").strip()
    
    if mode == '1':
        system.run_interactive_mode()
    
    elif mode == '2':
        profiles_file = input("사용자 프로필 JSON 파일 경로: ").strip()
        if os.path.exists(profiles_file):
            system.batch_processing_mode(profiles_file)
        else:
            print(f"❌ 프로필 파일을 찾을 수 없습니다: {profiles_file}")
    
    elif mode == '3':
        create_sample_user_profiles()
        print("\n샘플 프로필 생성 후 배치 처리 모드를 실행하시겠습니까?")
        if input("(y/n): ").strip().lower() == 'y':
            system.batch_processing_mode('sample_user_profiles.json')
    
    else:
        print("❌ 올바른 선택이 아닙니다.")


if __name__ == "__main__":
    main()