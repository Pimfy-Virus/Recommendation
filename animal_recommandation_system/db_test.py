import pymysql, ssl, os

conn = pymysql.connect(
    host='mixbuddy-rds.ct48ccoyknzp.ap-northeast-2.rds.amazonaws.com',
    port=3306,
    user='readonlyuser',
    password='K82bM6U7EB9SQi',
    db='pimfy_homepage',
    ssl={'ca': os.path.expanduser('~/rds-combined-ca-bundle.pem')}
)

with conn.cursor() as cur:
    cur.execute("SHOW TABLES;")
    print(cur.fetchall())
