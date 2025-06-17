import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# 크롤링 할 핌피바이러스 사이트
BASE_LIST_URL = "https://www.pimfyvirus.com/search/01"
BASE_DOMAIN = "https://www.pimfyvirus.com"

# --- 리스트 페이지 HTML을 가져오는 함수 ---
def get_list_page(page_num):
    """
    지정된 페이지 번호에 해당하는 리스트 페이지의 HTML을 가져옵니다.
    페이지네이션 URL 형식: /search/01/p={page_num}/
    """
    # 1페이지는 /search/01/ 이고, 2페이지부터는 /search/01/p=2/ 형태
    url = f"{BASE_LIST_URL}/p={page_num}/" if page_num > 1 else BASE_LIST_URL + "/"
    try:
        res = requests.get(url)
        res.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return BeautifulSoup(res.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"오류: 페이지 {page_num} 로드 실패: {e}")
        return None

# --- 상세 페이지 데이터를 추출하는 함수 ---
def get_detail_data(detail_url):
    """
    주어진 상세 페이지 URL에서 필요한 데이터를 추출합니다.
    """
    data = {}
    try:
        res = requests.get(detail_url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # 공고 번호 추출
        num_paragraph = soup.find('p', class_='num')
        if num_paragraph:
            announcement_text = num_paragraph.get_text(strip=True)
            data["공고번호"] = announcement_text.replace("공고 번호 :", "").strip()
        else:
            data["공고번호"] = None

        # 기본 정보 (info_box) 추출
        info_box = soup.select_one(".info_box")
        if info_box:
            dts = info_box.find_all('dt')
            dds = info_box.find_all('dd')
            if dts and dds and len(dts) == len(dds):
                for dt, dd in zip(dts, dds):
                    key = dt.get_text(strip=True).replace(":", "")
                    value = dd.get_text(strip=True)
                    data[key] = value
            else:
                for line in info_box.get_text(separator="|").split("|"):
                    if ':' in line:
                        k, v = line.split(":", 1)
                        data[k.strip()] = v.strip()

        # 해시태그 추출
        tag_wrap = soup.select_one(".tag_wrap")
        if tag_wrap:
            hashtags_list = [p.get_text(strip=True) for p in tag_wrap.select("p")]
            data['해시태그'] = ", ".join(hashtags_list)
        else:
            data['해시태그'] = None

        # 메인 테이블 정보 추출 (동물 상세 정보)
        tb_wrap_div_main = soup.select_one("div.tb_wrap")
        if tb_wrap_div_main:
            rows = tb_wrap_div_main.select("table tr")
            for row in rows:
                header_th = row.select_one("th")
                value_td = row.select_one("td")
                if header_th and value_td:
                    key_name = header_th.get_text(strip=True).replace(":", "").strip()
                    value_data = value_td.get_text(strip=True)
                    data[key_name] = value_data

        # '임보 조건' 섹션 데이터 추출
        view_main_div = soup.select_one("div.view_main")
        if view_main_div:
            im_bo_condition_line_div = None
            for l_div in view_main_div.select("div.line"):
                tit_p_in_line = l_div.select_one("p.tit")
                if tit_p_in_line and "임보 조건" in tit_p_in_line.get_text(strip=True):
                    im_bo_condition_line_div = l_div
                    break
            
            if im_bo_condition_line_div:
                tb_wrap_div_condition = im_bo_condition_line_div.select_one("div.tb_wrap")
                if tb_wrap_div_condition:
                    condition_rows = tb_wrap_div_condition.select("table tr")
                    for row in condition_rows:
                        header_th = row.select_one("th")
                        value_td = row.select_one("td")
                        if header_th and value_td:
                            key_name = header_th.get_text(strip=True).replace(":", "").strip()
                            value_data = value_td.get_text(strip=True)
                            data[f"임보조건_{key_name}"] = value_data 

        # '이런 집도 가능해요' 데이터 추출
        possible_homes = []
        box_pt40_div = soup.select_one("div.box.pt40")
        if box_pt40_div:
            check_items = box_pt40_div.select("div.check_i.a") 
            for item_div in check_items:
                text_content = item_div.get_text(strip=True)
                if text_content:
                    possible_homes.append(text_content)
        data["이런_집도_가능해요"] = ", ".join(possible_homes) if possible_homes else None

        # '건강 정보' 섹션 데이터 추출
        if view_main_div:
            health_line_div = None
            for l_div in view_main_div.select("div.line"):
                tit_p_in_line = l_div.select_one("p.tit")
                if tit_p_in_line and "건강 정보" in tit_p_in_line.get_text(strip=True):
                    health_line_div = l_div
                    break
            
            if health_line_div:
                health_tb_wrap_div = health_line_div.select_one("div.tb_wrap")
                if health_tb_wrap_div:
                    health_rows = health_tb_wrap_div.select("table tr")
                    for row in health_rows:
                        header_th = row.select_one("th")
                        value_td = row.select_one("td")
                        if header_th and value_td:
                            key_name = header_th.get_text(strip=True).replace(":", "").strip()
                            value_data = value_td.get_text(strip=True)
                            data[f"건강정보_{key_name}"] = value_data 
        
        # '참고용 정보' 섹션 데이터 추출
        if view_main_div:
            reference_line_div = None
            for l_div in view_main_div.select("div.line"):
                tit_p_in_line = l_div.select_one("p.tit")
                if tit_p_in_line and "참고용 정보" in tit_p_in_line.get_text(strip=True):
                    reference_line_div = l_div
                    break
            
            if reference_line_div:
                ref_box_div = reference_line_div.select_one("div.box")
                if ref_box_div:
                    ref_items = ref_box_div.select("ul li")
                    for li_tag in ref_items:
                        column_p = li_tag.select_one("p.t_bef")
                        data_p = li_tag.select_one("p.a")
                        if column_p and data_p:
                            key_name = column_p.get_text(strip=True)
                            value_data = data_p.get_text(strip=True)
                            data[f"참고용정보_{key_name}"] = value_data
        
        # '책임자 제공 사항' 섹션 데이터 추출
        if view_main_div:
            responsible_line_div = None
            for l_div in view_main_div.select("div.line"):
                tit_p_in_line = l_div.select_one("p.tit")
                if tit_p_in_line and "책임자 제공 사항" in tit_p_in_line.get_text(strip=True):
                    responsible_line_div = l_div
                    break
            
            if responsible_line_div:
                responsible_box_div = responsible_line_div.select_one("div.box.pt40")
                if responsible_box_div:
                    checked_items = responsible_box_div.select("div.check_i.a") 
                    responsible_provisions = []
                    for item_div in checked_items:
                        text_content = item_div.get_text(strip=True)
                        if text_content:
                            responsible_provisions.append(text_content)
                    data["책임자_제공_사항"] = ", ".join(responsible_provisions) if responsible_provisions else None

    except requests.exceptions.RequestException as e:
        print(f"오류: 상세 페이지 로드 실패: {detail_url} - {e}")
    except Exception as e:
        print(f"오류: 상세 페이지 파싱 실패: {detail_url} - {e}")
    return data

# --- 리스트 페이지에서 동물 항목들을 추출하는 함수 ---
def get_list_items(soup):
    """
    리스트 페이지의 BeautifulSoup 객체에서 각 동물 항목의 상세 링크와 기본 정보를 추출합니다.
    추출된 상세 링크를 통해 상세 데이터를 추가로 가져옵니다.
    """
    items = []
    # 리스트에 있는 개별 동물 항목들을 선택
    lis = soup.select("div.lst_wrap ul li")

    if not lis:
        print("현재 페이지에서 동물 항목을 찾을 수 없습니다.")
        return items

    for li in lis:
        item = {}
        anchor = li.select_one("a") # 각 항목의 상세 페이지 링크
        if not anchor:
            continue

        href = anchor.get("href")
        full_url = BASE_DOMAIN + href
        item["상세링크"] = full_url

        # 이름 추출
        top = anchor.select_one(".top")
        if top:
            top_text = top.get_text(strip=True)
            parts = top_text.split(maxsplit=1)
            item["이름"] = parts[0] if parts else None
        else:
            item["이름"] = None

        # 공고번호 추출
        bottom = anchor.select_one(".bottom")
        if bottom:
            for part in bottom.get_text(separator="|").split("|"):
                part = part.strip()
                if "공고번호" in part:
                    item["공고번호"] = part.replace("공고번호", "").strip()
        else:
            item["공고번호"] = None

        try:
            # 상세 페이지에서 추가 데이터 추출 및 병합
            detail_data = get_detail_data(full_url)
            item.update(detail_data)

        except Exception as e:
            print(f"상세 페이지 크롤링 실패: {full_url} – {e}")

        items.append(item)
        time.sleep(0.3) # 서버 부하를 줄이기 위한 딜레이

    return items

# 메인 크롤링 루프
all_data = []
PAGES_TO_CRAWL = 100 # 크롤링할 최대 페이지 수


print(f"--- 크롤링 시작 (총 {PAGES_TO_CRAWL} 페이지) ---")

for p in range(1, PAGES_TO_CRAWL + 1):
    print(f"▶ 페이지 {p} 크롤링 중...")
    soup_list_page = get_list_page(p)
# 간단한 페이지 디버깅
    if soup_list_page is None:
        print(f"페이지 {p} 로드에 실패하여 다음 페이지로 넘어갑니다.") 
        continue

    page_items = get_list_items(soup_list_page)

    if not page_items:
        print(f"페이지 {p} 에서 더 이상 동물 항목을 찾을 수 없습니다. 크롤링을 종료합니다.")
        break

    all_data.extend(page_items)

print(f"\n--- 크롤링 완료! 총 {len(all_data)}개 데이터 수집 ---")

# Pandas DataFrame으로 변환
df = pd.DataFrame(all_data)

# DataFrame 출력 설정 (모든 열 표시, 너비 확장)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n--- 수집된 데이터 (DataFrame) ---")
print(df)

# CSV 파일로 저장 
df.to_csv("pimfyvirus_dog_data.csv", index=False, encoding="utf-8-sig")
print("\n데이터가 'pimfyvirus_dog_data.csv' 파일로 저장되었습니다.")
