"""
임시보호 동물 필터링 모듈
사용자 조건에 따라 동물을 필터링하는 기능 제공
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import re


class AnimalFilter:
    """임시보호 동물 필터링 클래스"""
    
    def __init__(self, animals: Optional[pd.DataFrame] = None):
        self.animals = animals if animals is not None else pd.DataFrame()
        self.filtered_results = pd.DataFrame()
    
    def set_animals(self, animals: pd.DataFrame) -> 'AnimalFilter':
        """동물 데이터 설정"""
        self.animals = animals
        return self
    
    def apply_filters(self, filter_criteria: Dict) -> pd.DataFrame:
        """
        복합 필터 적용
        
        Args:
            filter_criteria: 필터링 조건 딕셔너리
            
        Returns:
            필터링된 동물 데이터프레임
        """
        if self.animals.empty:
            return pd.DataFrame()
        
        # 기본 상태 필터 (임보가능한 동물만)
        results = self.animals[self.animals['status'] == '임보가능'].copy()
        
        # 각 필터 조건 적용
        if 'region' in filter_criteria and filter_criteria['region']:
            results = self._filter_by_region(results, filter_criteria['region'])
        
        if 'gender' in filter_criteria and filter_criteria['gender']:
            results = self._filter_by_gender(results, filter_criteria['gender'])
        
        if 'care_type' in filter_criteria and filter_criteria['care_type']:
            results = self._filter_by_care_type(results, filter_criteria['care_type'])
        
        if 'age_range' in filter_criteria and filter_criteria['age_range']:
            results = self._filter_by_age_range(results, filter_criteria['age_range'])
        
        if 'weight_range' in filter_criteria and filter_criteria['weight_range']:
            results = self._filter_by_weight_range(results, filter_criteria['weight_range'])
        
        if 'neutered' in filter_criteria and filter_criteria['neutered'] is not None:
            results = self._filter_by_neutered(results, filter_criteria['neutered'])
        
        if 'hashtags' in filter_criteria and filter_criteria['hashtags']:
            results = self._filter_by_hashtags(results, filter_criteria['hashtags'])
        
        if 'suitable_homes' in filter_criteria and filter_criteria['suitable_homes']:
            results = self._filter_by_suitable_homes(results, filter_criteria['suitable_homes'])
        
        if 'behavior_traits' in filter_criteria and filter_criteria['behavior_traits']:
            results = self._filter_by_behavior_traits(results, filter_criteria['behavior_traits'])
        
        if 'health_requirements' in filter_criteria and filter_criteria['health_requirements']:
            results = self._filter_by_health_requirements(results, filter_criteria['health_requirements'])
        
        if 'care_preferences' in filter_criteria and filter_criteria['care_preferences']:
            results = self._filter_by_care_preferences(results, filter_criteria['care_preferences'])
        
        self.filtered_results = results
        return results
    
    def _filter_by_region(self, animals: pd.DataFrame, regions: Union[str, List[str]]) -> pd.DataFrame:
        """지역별 필터링"""
        if isinstance(regions, str):
            regions = [regions]
        
        return animals[
            animals['rescue_location'].isin(regions) | 
            animals['care_conditions'].apply(lambda x: x.get('region') == '전국' if isinstance(x, dict) else False)
        ]
    
    def _filter_by_gender(self, animals: pd.DataFrame, genders: Union[str, List[str]]) -> pd.DataFrame:
        """성별 필터링"""
        if isinstance(genders, str):
            genders = [genders]
        return animals[animals['gender'].isin(genders)]
    
    def _filter_by_care_type(self, animals: pd.DataFrame, care_types: Union[str, List[str]]) -> pd.DataFrame:
        """임보 종류 필터링"""
        if isinstance(care_types, str):
            care_types = [care_types]
        return animals[animals['care_type'].isin(care_types)]
    
    def _filter_by_age_range(self, animals: pd.DataFrame, age_range: Dict[str, int]) -> pd.DataFrame:
        """나이 범위 필터링"""
        min_age = age_range.get('min', 0)
        max_age = age_range.get('max', 100)
        
        return animals[
            (animals['age'] >= min_age) & 
            (animals['age'] <= max_age) & 
            animals['age'].notna()
        ]
    
    def _filter_by_weight_range(self, animals: pd.DataFrame, weight_range: Dict[str, float]) -> pd.DataFrame:
        """몸무게 범위 필터링"""
        min_weight = weight_range.get('min', 0)
        max_weight = weight_range.get('max', 100)
        
        return animals[
            (animals['weight'] >= min_weight) & 
            (animals['weight'] <= max_weight) & 
            animals['weight'].notna()
        ]
    
    def _filter_by_neutered(self, animals: pd.DataFrame, neutered: bool) -> pd.DataFrame:
        """중성화 여부 필터링"""
        return animals[animals['neutered'] == neutered]
    
    def _filter_by_hashtags(self, animals: pd.DataFrame, required_hashtags: List[str]) -> pd.DataFrame:
        """해시태그 필터링 (OR 조건)"""
        def has_matching_hashtag(hashtags):
            if not hashtags:
                return False
            return any(
                any(tag in animal_tag or animal_tag in tag for animal_tag in hashtags)
                for tag in required_hashtags
            )
        
        return animals[animals['hashtags'].apply(has_matching_hashtag)]
    
    def _filter_by_suitable_homes(self, animals: pd.DataFrame, home_types: List[str]) -> pd.DataFrame:
        """적합한 가정 유형 필터링"""
        def matches_home_type(care_conditions):
            if not isinstance(care_conditions, dict) or 'suitable_homes' not in care_conditions:
                return False
            suitable_homes = care_conditions['suitable_homes']
            if not suitable_homes:
                return False
            return any(
                any(home_type in suitable_home or suitable_home in home_type 
                    for suitable_home in suitable_homes)
                for home_type in home_types
            )
        
        return animals[animals['care_conditions'].apply(matches_home_type)]
    
    def _filter_by_behavior_traits(self, animals: pd.DataFrame, trait_requirements: Dict) -> pd.DataFrame:
        """행동 특성 필터링"""
        def meets_behavior_requirements(behavior_traits):
            if not isinstance(behavior_traits, dict):
                return False
            
            for trait_name, requirement in trait_requirements.items():
                animal_value = behavior_traits.get(trait_name)
                
                if animal_value is None:
                    continue
                
                # 범위 조건 (min, max)
                if 'min' in requirement and animal_value < requirement['min']:
                    return False
                if 'max' in requirement and animal_value > requirement['max']:
                    return False
                
                # 정확한 값 조건
                if 'exact' in requirement and animal_value != requirement['exact']:
                    return False
            
            return True
        
        return animals[animals['behavior_traits'].apply(meets_behavior_requirements)]
    
    def _filter_by_health_requirements(self, animals: pd.DataFrame, health_reqs: Dict) -> pd.DataFrame:
        """건강 요구사항 필터링"""
        def meets_health_requirements(health_info):
            if not isinstance(health_info, dict):
                return False
            
            # 예방접종 완성도 확인
            if 'min_vaccinations' in health_reqs and health_info.get('vaccination'):
                vaccination_count = len(health_info['vaccination'])
                if vaccination_count < health_reqs['min_vaccinations']:
                    return False
            
            # 병력이 없는 동물만 원하는 경우
            if health_reqs.get('no_medical_history', False) and health_info.get('medical_history'):
                return False
            
            # 특정 질병 제외
            if 'exclude_conditions' in health_reqs and health_info.get('medical_history'):
                medical_history = str(health_info['medical_history']).lower()
                has_excluded_condition = any(
                    condition.lower() in medical_history
                    for condition in health_reqs['exclude_conditions']
                )
                if has_excluded_condition:
                    return False
            
            return True
        
        return animals[animals['health_info'].apply(meets_health_requirements)]
    
    def _filter_by_care_preferences(self, animals: pd.DataFrame, care_prefs: Dict) -> pd.DataFrame:
        """임보 조건 선호도 필터링"""
        def meets_care_preferences(care_conditions):
            if not isinstance(care_conditions, dict):
                return False
            
            # 임보 기간 조건
            if 'max_duration' in care_prefs and care_conditions.get('duration'):
                if care_conditions['duration'] > care_prefs['max_duration']:
                    return False
            
            # 픽업 방식 조건
            if 'pickup_method' in care_prefs and care_conditions.get('pickup'):
                if care_prefs['pickup_method'] not in care_conditions['pickup']:
                    return False
            
            # 추가 조건 제외 사항
            if 'exclude_conditions' in care_prefs and care_conditions.get('additional_conditions'):
                additional_conditions = str(care_conditions['additional_conditions']).lower()
                has_excluded_condition = any(
                    condition.lower() in additional_conditions
                    for condition in care_prefs['exclude_conditions']
                )
                if has_excluded_condition:
                    return False
            
            return True
        
        return animals[animals['care_conditions'].apply(meets_care_preferences)]
    
    def apply_soft_filtering(self, preferences: Dict, threshold: float = 0.3) -> pd.DataFrame:
        """
        점수 기반 소프트 필터링 (추천 시스템용)
        
        Args:
            preferences: 사용자 선호도 (가중치 포함)
            threshold: 최소 점수 임계값
            
        Returns:
            점수순으로 정렬된 동물 데이터프레임
        """
        available_animals = self.animals[self.animals['status'] == '임보가능'].copy()
        
        if available_animals.empty:
            return pd.DataFrame()
        
        # 각 동물에 대해 매치 점수 계산
        match_scores = []
        for idx, animal in available_animals.iterrows():
            score = self._calculate_match_score(animal, preferences)
            match_scores.append(score)
        
        available_animals['match_score'] = match_scores
        
        # 임계값 이상인 동물만 필터링하고 점수순으로 정렬
        filtered_animals = available_animals[available_animals['match_score'] >= threshold]
        self.filtered_results = filtered_animals.sort_values('match_score', ascending=False)
        
        return self.filtered_results
    
    def _calculate_match_score(self, animal: pd.Series, preferences: Dict) -> float:
        """동물과 사용자 선호도 간 매치 점수 계산"""
        total_score = 0
        total_weight = 0
        
        weights = preferences.get('weights', {})
        
        # 지역 매치
        if 'region' in preferences:
            weight = weights.get('region', 1)
            score = self._calculate_region_score(animal, preferences['region'])
            total_score += score * weight
            total_weight += weight
        
        # 나이 매치
        if 'age_preference' in preferences:
            weight = weights.get('age', 1)
            score = self._calculate_age_score(animal, preferences['age_preference'])
            total_score += score * weight
            total_weight += weight
        
        # 크기 매치
        if 'size_preference' in preferences:
            weight = weights.get('size', 1)
            score = self._calculate_size_score(animal, preferences['size_preference'])
            total_score += score * weight
            total_weight += weight
        
        # 성격 매치
        if 'personality_traits' in preferences:
            weight = weights.get('personality', 1)
            score = self._calculate_personality_score(animal, preferences['personality_traits'])
            total_score += score * weight
            total_weight += weight
        
        # 행동 특성 매치
        if 'behavior_preferences' in preferences:
            weight = weights.get('behavior', 1)
            score = self._calculate_behavior_score(animal, preferences['behavior_preferences'])
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_region_score(self, animal: pd.Series, preferred_regions: List[str]) -> float:
        """지역 점수 계산"""
        care_conditions = animal.get('care_conditions', {})
        if isinstance(care_conditions, dict) and care_conditions.get('region') == '전국':
            return 1.0
        if animal.get('rescue_location') in preferred_regions:
            return 1.0
        return 0.0
    
    def _calculate_age_score(self, animal: pd.Series, age_preference: Dict) -> float:
        """나이 점수 계산"""
        age = animal.get('age')
        if pd.isna(age):
            return 0.5  # 나이 불명인 경우 중간 점수
        
        preferred = age_preference.get('preferred', {})
        acceptable = age_preference.get('acceptable', {})
        
        if preferred.get('min', 0) <= age <= preferred.get('max', 100):
            return 1.0
        if acceptable.get('min', 0) <= age <= acceptable.get('max', 100):
            return 0.7
        
        return 0.0
    
    def _calculate_size_score(self, animal: pd.Series, size_preference: Dict) -> float:
        """크기 점수 계산"""
        weight = animal.get('weight')
        if pd.isna(weight):
            return 0.5
        
        preferred = size_preference.get('preferred', {})
        acceptable = size_preference.get('acceptable', {})
        
        if preferred.get('min', 0) <= weight <= preferred.get('max', 100):
            return 1.0
        if acceptable.get('min', 0) <= weight <= acceptable.get('max', 100):
            return 0.7
        
        return 0.0
    
    def _calculate_personality_score(self, animal: pd.Series, personality_traits: List[str]) -> float:
        """성격 점수 계산 (해시태그 기반)"""
        hashtags = animal.get('hashtags', [])
        if not hashtags:
            return 0.5
        
        matches = [
            trait for trait in personality_traits
            if any(trait in tag or tag in trait for tag in hashtags)
        ]
        
        return len(matches) / len(personality_traits) if personality_traits else 0.5
    
    def _calculate_behavior_score(self, animal: pd.Series, behavior_prefs: Dict) -> float:
        """행동 특성 점수 계산"""
        behavior_traits = animal.get('behavior_traits', {})
        if not isinstance(behavior_traits, dict):
            return 0.5
        
        total_score = 0
        valid_traits = 0
        
        for trait_name, preference in behavior_prefs.items():
            animal_value = behavior_traits.get(trait_name)
            if pd.isna(animal_value):
                continue
            
            ideal = preference.get('ideal')
            acceptable = preference.get('acceptable', [])
            
            if animal_value == ideal:
                total_score += 1.0
            elif animal_value in acceptable:
                total_score += 0.7
            else:
                # 거리 기반 점수 (1-5 스케일에서)
                distance = abs(animal_value - ideal) if ideal is not None else 0
                total_score += max(0, 1 - distance / 4)
            
            valid_traits += 1
        
        return total_score / valid_traits if valid_traits > 0 else 0.5
    
    def get_results(self) -> pd.DataFrame:
        """필터 결과 가져오기"""
        return self.filtered_results
    
    def get_result_stats(self) -> Dict:
        """필터 결과 통계"""
        if self.filtered_results.empty:
            return {'total_count': 0}
        
        results = self.filtered_results
        
        # 나이 분포 계산
        def categorize_age(age):
            if pd.isna(age):
                return '나이 불명'
            elif age <= 1:
                return '1세 이하'
            elif age <= 3:
                return '1-3세'
            elif age <= 7:
                return '4-7세'
            else:
                return '8세 이상'
        
        age_distribution = results['age'].apply(categorize_age).value_counts().to_dict()
        
        # 몸무게 분포 계산
        def categorize_weight(weight):
            if pd.isna(weight):
                return '몸무게 불명'
            elif weight < 5:
                return '소형견'
            elif weight <= 15:
                return '중형견'
            else:
                return '대형견'
        
        weight_distribution = results['weight'].apply(categorize_weight).value_counts().to_dict()
        
        return {
            'total_count': len(results),
            'gender_distribution': results['gender'].value_counts().to_dict(),
            'age_distribution': age_distribution,
            'weight_distribution': weight_distribution,
            'care_type_distribution': results['care_type'].value_counts().to_dict(),
            'region_distribution': results['rescue_location'].value_counts().head(10).to_dict()
        }
    
    def get_top_matches(self, n: int = 10) -> pd.DataFrame:
        """상위 매치 결과 반환"""
        if 'match_score' in self.filtered_results.columns:
            return self.filtered_results.head(n)
        else:
            return self.filtered_results.head(n)
    
    def export_results(self, filename: str):
        """결과를 CSV 파일로 내보내기"""
        if not self.filtered_results.empty:
            # 복잡한 구조를 평탄화하여 CSV로 저장
            export_data = []
            for idx, row in self.filtered_results.iterrows():
                export_row = {
                    'id': row.get('id'),
                    'name': row.get('name'),
                    'gender': row.get('gender'),
                    'age': row.get('age'),
                    'weight': row.get('weight'),
                    'care_type': row.get('care_type'),
                    'rescue_location': row.get('rescue_location'),
                    'hashtags': '|'.join(row.get('hashtags', [])),
                    'match_score': row.get('match_score', 0),
                    'detail_link': row.get('detail_link', '')
                }
                export_data.append(export_row)
            
            pd.DataFrame(export_data).to_csv(filename, index=False, encoding='utf-8')
            print(f"결과가 {filename}에 저장되었습니다.")
        else:
            print("저장할 결과가 없습니다.")


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터로 테스트
    sample_data = pd.DataFrame([
        {
            'id': '1',
            'name': '테스트독',
            'status': '임보가능',
            'gender': 'male',
            'age': 3,
            'weight': 8.5,
            'care_type': '일반임보',
            'rescue_location': '서울',
            'hashtags': ['애교쟁이', '사람좋아'],
            'care_conditions': {'region': '전국', 'duration': 3},
            'behavior_traits': {'affection': 4, 'human_friendly': 5}
        }
    ])
    
    # 필터 테스트
    filter_obj = AnimalFilter(sample_data)
    
    # 하드 필터링 테스트
    hard_filter_criteria = {
        'age_range': {'min': 1, 'max': 5},
        'weight_range': {'min': 5, 'max': 15}
    }
    
    hard_results = filter_obj.apply_filters(hard_filter_criteria)
    print(f"하드 필터링 결과: {len(hard_results)}마리")
    
    # 소프트 필터링 테스트
    soft_preferences = {
        'age_preference': {
            'preferred': {'min': 2, 'max': 4},
            'acceptable': {'min': 1, 'max': 6}
        },
        'personality_traits': ['애교쟁이', '사람좋아'],
        'weights': {'age': 1.5, 'personality': 1.0}
    }
    
    soft_results = filter_obj.apply_soft_filtering(soft_preferences)
    print(f"소프트 필터링 결과: {len(soft_results)}마리")
    
    # 결과 통계
    stats = filter_obj.get_result_stats()
    print(f"결과 통계: {stats}")