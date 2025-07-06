import pandas as pd
import pymysql
from datetime import datetime
from config import Config

class Database:
    def __init__(self):
        self.config = {
            'host': Config.DB_HOST,
            'port': Config.DB_PORT,
            'user': Config.DB_USER,
            'password': Config.DB_PASSWORD,
            'database': Config.DB_NAME,
            'charset': 'utf8mb4'
        }
    
    def get_all_data(self):
        """모든 동물 데이터 가져오기"""
        conn = pymysql.connect(**self.config)
        try:
            df = pd.read_sql("SELECT * FROM homeprotection", conn)
            print(f"📊 데이터 수집 완료: {len(df)}마리")
            return df
        finally:
            conn.close()
    
    def get_new_data(self, last_update):
        """새로운 데이터만 가져오기"""
        conn = pymysql.connect(**self.config)
        try:
            query = "SELECT * FROM homeprotection WHERE d_regis > %s"
            df = pd.read_sql(query, conn, params=[last_update])
            print(f"📊 신규 데이터: {len(df)}마리")
            return df
        finally:
            conn.close()