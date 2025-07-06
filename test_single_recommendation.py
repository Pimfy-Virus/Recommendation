#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 추천 테스트 스크립트
"""

import sys
import os
sys.path.append('/Users/sxxwings/git/pimfy/Recommendation/animal_recommandation_system')

from main import AnimalRecommendationMain

def main():
    print("🐕 단일 추천 테스트")
    print("=" * 50)
    
    # 추천 시스템 객체 생성
    recommender = AnimalRecommendationMain()
    
    # API 키 확인
    if not recommender.api_key or recommender.api_key == 'your-openai-api-key-here':
        print("❌ API 키가 설정되지 않았습니다.")
        return
    
    # 프로세서 초기화
    if not recommender.setup_processors():
        print("❌ 프로세서 초기화 실패")
        return
    
    # 시스템 로드
    if not recommender.load_system():
        print("❌ 시스템 로드 실패")
        return
    
    # 테스트 쿼리
    test_query = "활발하고 애교 많은 소형견을 원해요"
    
    try:
        # 단일 추천 테스트 실행
        success = recommender.test_single_recommendation(test_query)
        
        if success:
            print("\n🎉 추천 테스트 성공!")
        else:
            print("\n❌ 추천 테스트 실패")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
