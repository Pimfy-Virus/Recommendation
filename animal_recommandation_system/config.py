import os
from dataclasses import dataclass

@dataclass
class Config:
    # Database
    DB_HOST = "mixbuddy-rds.ct48ccoyknzp.ap-northeast-2.rds.amazonaws.com"
    DB_PORT = 3306
    DB_USER = "readonlyuser"
    DB_PASSWORD = "K82bM6U7EB9SQi"
    DB_NAME = "pimfy_homepage"
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = "text-embedding-3-large"
    
    # Files
    DATA_FILE = "data/animals.pkl"
    EMBEDDING_FILE = "data/embeddings.pkl"