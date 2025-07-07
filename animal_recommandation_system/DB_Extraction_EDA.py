import pandas as pd
from sqlalchemy import create_engine, inspect
from typing import Optional, Dict, List
from typing import Optional, List
from datetime import datetime
   
    
###동적으로 DB에서 데이터 추출 및 전처리하는 클래스
    
class DBLoader:

    def __init__(self, 
                 host: str = "mixbuddy-rds.ct48ccoyknzp.ap-northeast-2.rds.amazonaws.com",
                 port: int = 3306,
                 dbname: str = "pimfy_homepage",
                 user: str = "readonlyuser",
                 password: str = "K82bM6U7EB9SQi"):
        self.conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.conn_str)
        self.table_names = ["homeprotection", "homeprotectionsub01", "homeprotectionsub04"]
        print(f"DB 연결 엔진 생성 완료 ({dbname})")
        
    def load_table(self, table_name: str) -> pd.DataFrame:
        # homeprotection 쿼리 조건 분리-> state가 입양 가능으로 행들만 추출
        if table_name == "homeprotection":
            query = f"SELECT * FROM {table_name} WHERE state='임보가능';"
        else:
            query = f"SELECT * FROM {table_name};"
        
        try:
            print(f"{table_name} 데이터 로드 중...")
            df = pd.read_sql(query, con=self.engine)
            print(f"{table_name} 로드 완료: {len(df):,} rows")
            return df
        except Exception as e:
            print(f"{table_name} 로드 실패(DB연결 확인 필요): {e}")
            return pd.DataFrame()
#--------------------------------- 전처리-------------------------------------------------
    def EDA(self) -> pd.DataFrame:
        #조건부 테이블 로드
        dfs = [self.load_table(table) for table in self.table_names]

        #  병합 방식 
        base_df = dfs[0]  # homeprotection
        sub01_df = dfs[1] # homeprotectionsub01
        sub04_df = dfs[2] # homeprotectionsub04

        # 병합 방식  2
    
        merged_df = pd.merge(base_df, sub01_df, left_on="uid", right_on="uid", how="left")
        merged_df = pd.merge(merged_df, sub04_df, left_on="uid", right_on="uid", how="left")

        print(f"\n 병합 완료: {len(merged_df):,} rows")
       
        # 2️ 필요없는 컬럼 삭제
        columns_to_drop = [
            "ordernum", "subject", "addinfo02", "inlocation", "mbuid", "d_regis",
            "name", "puid_x", "puid_y", "type", "addinfo01_y", "addinfo06", "addinfo20",
            "addinfo02_x", "kind", "steplast", "hit", "addinfo09", "addinfo23"
        ]
        merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])
    

        # 3️ 컬럼명 변경
        rename_map = {
            'uid': 'Pk',
            'state': '상태',
            'addinfo01_x': '이름',
            'addinfo07': '몸무게',
            'addinfo03_x': '성별',
            'addinfo04': '중성화여부',
            'addinfo05': '출생',
            'addinfo08': '해시태그',
            'addinfo10': '성격및특징',
            'addinfo11': '건강_기타사항',
            'addinfo12': '임보가능_지역',
            'addinfo13': 'addinfo13',
            'addinfo13sub01': 'addinfo13sub01',
            'addinfo13sub02': '임보가능_기간',
            'addinfo14': '픽업',
            'addinfo15': '이런집도가능',
            'addinfo16': '필수조건',
            'addinfo17': '접종현황',
            'addinfo18': '검사현황',
            'addinfo19': '병력사항',
            'addinfo21': '책임자제공사항_검토필요',
            'addinfo22': '책임_기타사항',
            's_pic01': '사진',
            'snsinfo': 'sns',
            'addinfo03_y': '유의사항'
        }
        merged_df = merged_df.rename(columns={k: v for k, v in rename_map.items() if k in merged_df.columns})
    

        # 4️ addinfo13, addinfo13sub01, 임보가능_기간 병합
        merge_columns = ['addinfo13', 'addinfo13sub01', '임보가능_기간']
        available_cols = [col for col in merge_columns if col in merged_df.columns]
        if available_cols:
            merged_df["임보가능_기간_통합"] = (
                merged_df[available_cols]
                .astype(str)
                .replace("nan", "")
                .agg(" ".join, axis=1)
                .str.strip()
                .replace("", pd.NA)
            )
            merged_df = merged_df.drop(columns=available_cols)
            print(f"컬럼 병합 완료 -> '임보가능_기간_통합' 생성")
         #5 나이가 연도+ 추정(없을수도 있음)으로 나와서 현재 날짜 기준으로 빼고 추정이란 단어가 있으면 붙이는 식으로 나이를 전처리
        today = datetime.today()

        if "출생" in merged_df.columns:
            def calculate_age(value):
                if pd.isna(value):
                    return pd.NA
                value_str = str(value)
                has_estimate = "추정" in value_str
                value_str_clean = value_str.replace("추정", "")
                try:
                    year = int(value_str_clean[:4])
                    # 월 정보가 있으면 추출, 없으면 1월로 기본 처리
                    if len(value_str_clean) >= 6:
                        month = int(value_str_clean[4:6])
                    else:
                        month = 1
                    birth_date = datetime(year, month, 1)
                    age_in_months = (today.year - birth_date.year) * 12 + (today.month - birth_date.month)
                    age_str = f"{age_in_months}개월"
                    if has_estimate:
                        age_str += "(추정)"
                    return age_str
                except Exception:
                    return pd.NA
            merged_df["나이_개월수"] = merged_df["출생"].apply(calculate_age)
           

        
        #6 Pk 컬럼 인덱스화
        if "Pk" in merged_df.columns:
            merged_df = merged_df.set_index("Pk").reset_index()
          

        return merged_df


## 간단하게 확인해보기
#db_loader = DBLoader()

# 
#merged_df = db_loader.EDA()


#merged_df.head()
