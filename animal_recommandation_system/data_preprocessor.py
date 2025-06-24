"""
임시보호 동물 데이터 전처리 모듈
CSV 데이터를 정제하고 필터링을 위한 구조화된 형태로 변환
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """임시보호 동물 데이터 전처리 클래스"""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.metadata = {}
    
    def load_and_process(self, csv_path: str) -> pd.DataFrame:
        """
        CSV 파일을 로드하고 기본 전처리 수행
        
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            전처리된 데이터프레임
        """
        # CSV 파일 로드
        self.raw_data = pd.read_csv(csv_path, encoding='utf-8')
        print(f"총 {len(self.raw_data)}개의 동물 데이터 로드됨")
        
        # 데이터 전처리 수행
        self.processed_data = self._process_all_data()
        
        # 메타데이터 생성
        self._generate_metadata()
        
        print(f"전처리 완료: {len(self.processed_data)}개의 데이터 처리됨")
        return self.processed_data
    
    def _process_all_data(self) -> pd.DataFrame:
        """모든 데이터에 대해 전처리 수행"""
        processed_list = []
        
        for idx, row in self.raw_data.iterrows():
            processed_animal = self._process_animal_data(row)
            processed_list.append(processed_animal)
        
        return pd.DataFrame(processed_list)
    
    def _process_animal_data(self, animal: pd.Series) -> Dict:
        """개별 동물 데이터 전처리"""
        return {
            # 기본 정보
            'id': self._extract_id(animal.get('상세링크', '')),
            'name': str(animal.get('이름', '')).strip(),
            'status': str(animal.get('현 상황', '')).strip(),
            'care_type': str(animal.get('임보종류', '')).strip(),
            'rescue_location': str(animal.get('구조 지역', '')).strip(),
            
            # 동물 기본 특성
            'gender': self._normalize_gender(animal.get('성별')),
            'neutered': self._normalize_neutered(animal.get('중성화 여부')),
            'birth_year': self._extract_birth_year(animal.get('출생시기')),
            'weight': self._extract_weight(animal.get('몸무게')),
            'age': self._calculate_age(animal.get('출생시기')),
            
            # 해시태그 처리
            'hashtags': self._process_hashtags(animal.get('해시태그')),
            
            # 임보 조건
            'care_conditions': {
                'region': str(animal.get('임보조건_지역', '')).strip(),
                'duration': self._process_duration(animal.get('임보조건_임보 기간')),
                'pickup': str(animal.get('임보조건_픽업', '')).strip(),
                'additional_conditions': animal.get('임보조건_기타 조건'),
                'suitable_homes': self._process_suitable_homes(animal.get('이런_집도_가능해요'))
            },
            
            # 건강 정보
            'health_info': {
                'vaccination': self._process_vaccination(animal.get('건강정보_접종 현황')),
                'examination': animal.get('건강정보_검사 현황'),
                'medical_history': animal.get('건강정보_병력 사항'),
                'additional_notes': animal.get('건강정보_기타 사항')
            },
            
            # 행동 특성 (1-5 스케일)
            'behavior_traits': {
                'toilet_training': self._safe_int_convert(animal.get('참고용정보_배변')),
                'walking_needs': self._safe_int_convert(animal.get('참고용정보_산책')),
                'barking': self._safe_int_convert(animal.get('참고용정보_짖음')),
                'separation_anxiety': self._safe_int_convert(animal.get('참고용정보_분리불안')),
                'shedding': self._safe_int_convert(animal.get('참고용정보_털빠짐')),
                'affection': self._safe_int_convert(animal.get('참고용정보_스킨십')),
                'human_friendly': self._safe_int_convert(animal.get('참고용정보_대인')),
                'dog_friendly': self._safe_int_convert(animal.get('참고용정보_대견')),
                'solo_living': self._safe_int_convert(animal.get('참고용정보_외동')),
                'cat_friendly': self._safe_int_convert(animal.get('참고용정보_대묘'))
            },
            
            # 기타 정보
            'support_provided': str(animal.get('책임자_제공_사항', '')).strip(),
            'detail_link': str(animal.get('상세링크', '')).strip(),
            'sns_link': animal.get('SNS'),
            'announcement_number': str(animal.get('공고번호', '')).strip()
        }
    
    def _extract_id(self, link: str) -> Optional[str]:
        """상세링크에서 ID 추출"""
        if not link or pd.isna(link):
            return None
        match = re.search(r'/(\d+)/$', str(link))
        return match.group(1) if match else None
    
    def _normalize_gender(self, gender) -> Optional[str]:
        """성별 정규화"""
        if pd.isna(gender):
            return None
        gender_str = str(gender).strip().lower()
        if '남' in gender_str or 'male' in gender_str:
            return 'male'
        elif '여' in gender_str or 'female' in gender_str:
            return 'female'
        return 'unknown'
    
    def _normalize_neutered(self, neutered) -> Optional[bool]:
        """중성화 여부 정규화"""
        if pd.isna(neutered):
            return None
        neutered_str = str(neutered).strip()
        return '완' in neutered_str or '완료' in neutered_str
    
    def _extract_birth_year(self, birth_info) -> Optional[int]:
        """출생연도 추출"""
        if pd.isna(birth_info):
            return None
        match = re.search(r'(\d{4})', str(birth_info))
        return int(match.group(1)) if match else None
    
    def _calculate_age(self, birth_info) -> Optional[int]:
        """나이 계산"""
        birth_year = self._extract_birth_year(birth_info)
        if birth_year is None:
            return None
        current_year = datetime.now().year
        return current_year - birth_year
    
    def _extract_weight(self, weight_str) -> Optional[float]:
        """몸무게 추출 (kg 단위로 변환)"""
        if pd.isna(weight_str):
            return None
        match = re.search(r'(\d+(?:\.\d+)?)', str(weight_str))
        return float(match.group(1)) if match else None
    
    def _process_hashtags(self, hashtag_str) -> List[str]:
        """해시태그 처리"""
        if pd.isna(hashtag_str):
            return []
        hashtags = str(hashtag_str).split(',')
        return [tag.replace('#', '').strip() for tag in hashtags if tag.strip()]
    
    def _process_duration(self, duration_str) -> Optional[int]:
        """임보 기간 처리"""
        if pd.isna(duration_str):
            return None
        match = re.search(r'(\d+)', str(duration_str))
        return int(match.group(1)) if match else None
    
    def _process_suitable_homes(self, homes_str) -> List[str]:
        """적합한 가정 유형 처리"""
        if pd.isna(homes_str):
            return []
        homes = str(homes_str).split(',')
        return [home.strip() for home in homes if home.strip()]
    
    def _process_vaccination(self, vaccination_str) -> Optional[List[Dict]]:
        """예방접종 정보 처리"""
        if pd.isna(vaccination_str):
            return None
        
        vaccinations = []
        lines = str(vaccination_str).split('\n')
        
        for line in lines:
            match = re.search(r'(\d+)차접종.*?(\d{2}\.\d{2}\.\d{2})', line)
            if match:
                vaccinations.append({
                    'round': int(match.group(1)),
                    'date': match.group(2)
                })
        
        return vaccinations if vaccinations else None
    
    def _safe_int_convert(self, value) -> Optional[int]:
        """안전한 정수 변환"""
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _generate_metadata(self):
        """메타데이터 생성 (필터링에 사용될 고유값들)"""
        if self.processed_data is None:
            return
        
        self.metadata = {
            'regions': self.processed_data['rescue_location'].dropna().unique().tolist(),
            'genders': self.processed_data['gender'].dropna().unique().tolist(),
            'care_types': self.processed_data['care_type'].dropna().unique().tolist(),
            'age_ranges': self._get_age_ranges(),
            'weight_ranges': self._get_weight_ranges(),
            'all_hashtags': self._get_all_hashtags(),
            'suitable_home_types': self._get_all_suitable_home_types()
        }
    
    def _get_age_ranges(self) -> List[Dict]:
        """나이 범위 생성"""
        ages = self.processed_data['age'].dropna()
        if ages.empty:
            return []
        
        return [
            {'label': '1세 미만', 'min': 0, 'max': 0},
            {'label': '1-3세', 'min': 1, 'max': 3},
            {'label': '4-7세', 'min': 4, 'max': 7},
            {'label': '8세 이상', 'min': 8, 'max': 100}
        ]
    
    def _get_weight_ranges(self) -> List[Dict]:
        """몸무게 범위 생성"""
        weights = self.processed_data['weight'].dropna()
        if weights.empty:
            return []
        
        return [
            {'label': '소형견 (5kg 미만)', 'min': 0, 'max': 4.9},
            {'label': '중형견 (5-15kg)', 'min': 5, 'max': 15},
            {'label': '대형견 (15kg 이상)', 'min': 15.1, 'max': 100}
        ]
    
    def _get_all_hashtags(self) -> List[str]:
        """모든 해시태그 수집"""
        all_tags = []
        for hashtags in self.processed_data['hashtags']:
            all_tags.extend(hashtags)
        return list(set(all_tags))
    
    def _get_all_suitable_home_types(self) -> List[str]:
        """모든 적합한 가정 유형 수집"""
        all_types = []
        for care_conditions in self.processed_data['care_conditions']:
            all_types.extend(care_conditions['suitable_homes'])
        return list(set(all_types))
    
    def get_processed_data(self) -> pd.DataFrame:
        """처리된 데이터 반환"""
        return self.processed_data
    
    def get_metadata(self) -> Dict:
        """메타데이터 반환"""
        return self.metadata
    
    def get_statistics(self) -> Dict:
        """데이터 통계 반환"""
        if self.processed_data is None:
            return None
        
        total_animals = len(self.processed_data)
        available_animals = len(self.processed_data[self.processed_data['status'] == '임보가능'])
        
        # 성별 분포
        gender_dist = self.processed_data['gender'].value_counts().to_dict()
        
        # 임보 종류 분포
        care_type_dist = self.processed_data['care_type'].value_counts().to_dict()
        
        # 지역 분포 (상위 10개)
        region_dist = self.processed_data['rescue_location'].value_counts().head(10).to_dict()
        
        # 평균 나이 및 몸무게
        avg_age = self.processed_data['age'].mean()
        avg_weight = self.processed_data['weight'].mean()
        
        return {
            'total': total_animals,
            'available': available_animals,
            'gender_distribution': gender_dist,
            'care_type_distribution': care_type_dist,
            'region_distribution': region_dist,
            'average_age': round(avg_age, 1) if not pd.isna(avg_age) else None,
            'average_weight': round(avg_weight, 1) if not pd.isna(avg_weight) else None
        }
    
    def save_processed_data(self, output_path: str):
        """전처리된 데이터를 파일로 저장"""
        if self.processed_data is not None:
            # DataFrame으로 변환하여 저장하기 위해 복잡한 구조를 평탄화
            flattened_data = []
            
            for _, row in self.processed_data.iterrows():
                flat_row = {
                    'id': row['id'],
                    'name': row['name'],
                    'status': row['status'],
                    'care_type': row['care_type'],
                    'rescue_location': row['rescue_location'],
                    'gender': row['gender'],
                    'neutered': row['neutered'],
                    'birth_year': row['birth_year'],
                    'weight': row['weight'],
                    'age': row['age'],
                    'hashtags': '|'.join(row['hashtags']) if row['hashtags'] else '',
                    
                    # 임보 조건
                    'care_region': row['care_conditions']['region'],
                    'care_duration': row['care_conditions']['duration'],
                    'care_pickup': row['care_conditions']['pickup'],
                    'care_additional_conditions': row['care_conditions']['additional_conditions'],
                    'suitable_homes': '|'.join(row['care_conditions']['suitable_homes']) if row['care_conditions']['suitable_homes'] else '',
                    
                    # 건강 정보
                    'vaccination_count': len(row['health_info']['vaccination']) if row['health_info']['vaccination'] else 0,
                    'medical_history': row['health_info']['medical_history'],
                    
                    # 행동 특성
                    'toilet_training': row['behavior_traits']['toilet_training'],
                    'walking_needs': row['behavior_traits']['walking_needs'],
                    'barking': row['behavior_traits']['barking'],
                    'separation_anxiety': row['behavior_traits']['separation_anxiety'],
                    'shedding': row['behavior_traits']['shedding'],
                    'affection': row['behavior_traits']['affection'],
                    'human_friendly': row['behavior_traits']['human_friendly'],
                    'dog_friendly': row['behavior_traits']['dog_friendly'],
                    'solo_living': row['behavior_traits']['solo_living'],
                    'cat_friendly': row['behavior_traits']['cat_friendly'],
                    
                    'detail_link': row['detail_link']
                }
                flattened_data.append(flat_row)
            
            pd.DataFrame(flattened_data).to_csv(output_path, index=False, encoding='utf-8')
            print(f"전처리된 데이터가 {output_path}에 저장되었습니다.")


# 사용 예시
if __name__ == "__main__":
    # 데이터 전처리기 초기화
    preprocessor = DataPreprocessor()
    
    # 데이터 로드 및 전처리
    processed_data = preprocessor.load_and_process('pimfyvirus_dog_data.csv')
    
    # 통계 정보 출력
    stats = preprocessor.get_statistics()
    print("\n=== 데이터 통계 ===")
    print(f"전체 동물 수: {stats['total']}")
    print(f"임보 가능한 동물 수: {stats['available']}")
    print(f"평균 나이: {stats['average_age']}세")
    print(f"평균 몸무게: {stats['average_weight']}kg")
    
    # 메타데이터 출력
    metadata = preprocessor.get_metadata()
    print(f"\n총 해시태그 종류: {len(metadata['all_hashtags'])}")
    print(f"구조 지역 수: {len(metadata['regions'])}")
    
    # 전처리된 데이터 저장
    preprocessor.save_processed_data('processed_animal_data.csv')