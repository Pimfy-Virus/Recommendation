"""
임시보호 동물 추천 시스템 설정 파일
"""

# 파일 경로 설정
DATA_PATH = 'pimfyvirus_dog_data.csv'
OUTPUT_DIR = 'results'
LOG_DIR = 'logs'

# 필터링 기본값
DEFAULT_FILTERS = {
    'status': '임보가능',
    'age_range': {'min': 0, 'max': 20},
    'weight_range': {'min': 0, 'max': 100}
}

# 추천 시스템 설정
RECOMMENDATION_SETTINGS = {
    'default_threshold': 0.3,
    'max_recommendations': 50,
    'default_weights': {
        'age': 1.0,
        'size': 1.0,
        'personality': 1.5,
        'behavior': 1.2,
        'region': 0.8,
        'health': 1.1
    }
}

# 행동 특성 매핑
BEHAVIOR_TRAIT_LABELS = {
    'toilet_training': '배변 훈련',
    'walking_needs': '산책 필요도',
    'barking': '짖음 정도',
    'separation_anxiety': '분리불안',
    'shedding': '털빠짐',
    'affection': '애정표현',
    'human_friendly': '사람 친화성',
    'dog_friendly': '개 친화성',
    'solo_living': '혼자 지내기',
    'cat_friendly': '고양이 친화성'
}

# 점수 척도 설명
SCORE_DESCRIPTIONS = {
    1: '매우 낮음/거의 없음',
    2: '낮음/가끔',
    3: '보통',
    4: '높음/자주',
    5: '매우 높음/항상'
}

# 나이 그룹 분류
AGE_GROUPS = [
    {'label': '퍼피 (1세 미만)', 'min': 0, 'max': 0},
    {'label': '어린 성견 (1-3세)', 'min': 1, 'max': 3},
    {'label': '성견 (4-7세)', 'min': 4, 'max': 7},
    {'label': '시니어 (8세 이상)', 'min': 8, 'max': 100}
]

# 크기 분류
SIZE_GROUPS = [
    {'label': '초소형견 (3kg 미만)', 'min': 0, 'max': 2.9},
    {'label': '소형견 (3-10kg)', 'min': 3, 'max': 10},
    {'label': '중형견 (10-25kg)', 'min': 10.1, 'max': 25},
    {'label': '대형견 (25kg 이상)', 'min': 25.1, 'max': 100}
]

# 임보 종류별 설명
CARE_TYPE_DESCRIPTIONS = {
    '일반임보': '일반적인 임시보호',
    '단기임보': '짧은 기간 임시보호',
    '입양전제': '입양을 전제로 한 임시보호',
    '긴급임보': '응급상황 임시보호',
    '릴레이임보': '여러 가정이 번갈아 임보',
    '수유임보': '새끼 강아지 수유 임보'
}

# 지역 그룹핑 (주요 지역)
MAJOR_REGIONS = [
    '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
    '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
]

# 성격 해시태그 그룹핑
PERSONALITY_GROUPS = {
    '활발한': ['활발', '에너지', '장난꾸러기', '꾸러기'],
    '차분한': ['조용', '차분', '얌전', '침착'],
    '사교적': ['사람좋아', '친화적', '사교적'],
    '애교': ['애교쟁이', '애교', '귀여운'],
    '똑똑한': ['똑똑이', '영리', '학습능력'],
    '순한': ['순둥이', '온순', '착한']
}

# 로깅 설정
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/system.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# UI 설정 (향후 웹 인터페이스용)
UI_SETTINGS = {
    'items_per_page': 10,
    'max_display_hashtags': 5,
    'thumbnail_size': (200, 200),
    'theme': 'light'
}

# API 설정 (향후 확장용)
API_SETTINGS = {
    'rate_limit': 100,  # 시간당 요청 수
    'cache_timeout': 3600,  # 1시간
    'max_results': 100
}