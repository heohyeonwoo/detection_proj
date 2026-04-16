# import random
# import pandas as pd
# import os
# import time

# def generate_5M_ultimate_drug_dataset(target_count=5000000):
#     print(f"==================================================")
#     print(f"🔥 [System] 글로벌/국내 총망라 마약 은어 500만 개 증강 엔진 가동 🔥")
#     print(f"==================================================")

#     # 1. 수사관님 제공(Reddit + 국내 하드코어) 총통합 마약 은어 사전
#     drug_slangs = [
#         # 필로폰/주사기
#         "아이스", "찬술", "시원한 술", "작대기", "빙두", "크리스탈", "눈꽃", "연필", "피로회복제", "가루", "크랭크",
#         # 코카인 (해외 Reddit 슬랭)
#         "블로우", "코크", "스노우", "하얀가루",
#         # 엑스터시/MDMA
#         "캔디", "몰리", "도리", "도리도리", "츄파춥스", "엠페러", "스컬", "몽키",
#         # 대마초/액상대마 (해외 Reddit 슬랭 포함)
#         "엠제이", "그래스", "위드", "떨", "고기", "풀", "브액", "덕트", "라바", "단풍", "허브",
#         # 케타민/환각버섯/LSD
#         "켓", "케이", "원석케이", "쉬룸", "우표", "엘시드", 
#         # 펜타닐 (신종)
#         "아파치", "하얀중국", "중국소녀", "댄스피버"
#     ]
    
#     # 2. 거래 규모 
#     amounts = ["한 잔", "반 잔", "1g", "0.5g", "시료", "샘플", "대량", "소량", "한 피스", "두 개"]
    
#     # 3. 범죄 조직원 역할
#     roles = ["지게꾼이", "드로퍼가", "상선 쪽에서", "배달원이", "작업조가"]
    
#     # 4. 행동 묘사
#     actions = [
#         "던지기 완료했습니다", "안전드랍 해뒀습니다", "좌표 찍어드립니다", 
#         "방금 숨겨놨습니다", "드랍 완료", "땅에 묻어뒀습니다", 
#         "비대면 세팅 끝났습니다", "사진이랑 좌표 보냈습니다", "테이프로 은밀하게 붙여놨습니다"
#     ]
    
#     # 5. 접두사 및 경고
#     prefixes = [
#         "입금 확인했습니다.", "사장님,", "오늘 바로,", "수량 넉넉합니다.", 
#         "확실하게,", "야당(끄나풀) 조심하시고,", "환자분들 주목,", "비밀보장 확실합니다."
#     ]
    
#     # 6. 구매자 문의 템플릿
#     questions = [
#         "혹시 오늘 바로 받을 수 있나요?", "직거래 말고 안전드랍 가능한가요?", 
#         "시원한거 있나요?", "드로퍼 배정됐나요?", "좌표 언제 나오나요?",
#         "고속도로(주사자국) 안 남는 걸로 있나요?", "도리도리 가능한가요?", "블로우 오늘 되나요?"
#     ]

#     # 7. 문장 조합 템플릿
#     templates = [
#         "{prefix} {location}에 {drug_slang} {amount} {action}.",
#         "{drug_slang} 구합니다. {location} 위주로 {action} 원해요.",
#         "{location} 좌표입니다. {drug_slang} 확인하고 텔레주세요.",
#         "{question} {drug_slang} {amount} 찾고 있습니다.",
#         "오늘 {drug_slang} 수량 확인 부탁드립니다. {role} {location}에 {action}.",
#         "{prefix} {location} 확인해 보세요. {role} {drug_slang} {action}.",
#         "{role} {location}에 {drug_slang} 안전드랍 완료했습니다. 좌표 확인하세요.",
#         "방금 {drug_slang} {amount} {location} 쪽에 묻어뒀습니다."
#     ]

#     # 8. [핵심] 다이나믹 좌표 생성기 (경우의 수를 수십억 개로 뻥튀기)
#     def get_dynamic_location():
#         loc_patterns = [
#             f"해운대역 {random.randint(1, 14)}번 출구 코인락커 {random.randint(1, 999)}번",
#             f"서면역 {random.randint(1, 15)}번 출구 지하 물품보관함 {random.randint(1, 500)}번",
#             f"강남역 {random.randint(1, 12)}번 출구 뒤편 도로 배전반",
#             f"아파트 {random.randint(101, 999)}동 {random.randint(1, 30)}층 소화전",
#             f"빌라 {random.randint(1, 10)}층 우편함 안쪽",
#             f"지하주차장 B{random.randint(1, 7)} 기둥 {random.choice(['A','B','C','D','E','F'])}{random.randint(1, 99)} 에어컨 실외기 밑",
#             f"상가 {random.randint(1, 15)}층 남자화장실 {random.randint(1, 10)}번째 칸 변기 뒤",
#             f"원룸 {random.randint(101, 1509)}호 통신단자함",
#             f"비상계단 {random.randint(2, 20)}층 소화기 안쪽",
#             f"무인택배함 {random.randint(1, 200)}번",
#             f"골목길 전봇대 고유번호 {random.randint(1000, 9999)} 철제 기둥 안쪽",
#             f"야산 둘레길 {random.randint(1, 50)}번째 나무 아래",
#             f"실외기 배관 틈새 {random.randint(1, 100)}구역",
#             f"아파트 복도 창고 설비 기구 {random.choice(['좌측', '우측', '뒷편'])}"
#         ]
#         return random.choice(loc_patterns)

#     generated_data = set() # 중복을 자동으로 걸러주는 자료구조
#     start_time = time.time()
    
#     print(f"[Process] 데이터 조합 시작! (목표: {target_count:,}건)")
    
#     # 9. 500만 건 무한 생성 루프
#     current_size = 0
#     while current_size < target_count:
#         template = random.choice(templates)
#         sentence = template.format(
#             prefix=random.choice(prefixes),
#             location=get_dynamic_location(), # 매번 숫자가 바뀌는 좌표
#             drug_slang=random.choice(drug_slangs),
#             amount=random.choice(amounts),
#             role=random.choice(roles),
#             action=random.choice(actions),
#             question=random.choice(questions)
#         )
        
#         generated_data.add(sentence)
#         new_size = len(generated_data)
        
#         # 10만 개 단위로 화면에 진행 상황 출력
#         if new_size > current_size and new_size % 100000 == 0:
#             elapsed = time.time() - start_time
#             print(f" ⏳ 진행 상황: {new_size:,} / {target_count:,} 건 생성 완료... (소요 시간: {elapsed:.1f}초)")
        
#         current_size = new_size

#     # 10. 메모리 최적화 변환 및 CSV 저장
#     print("\n[Process] 생성이 완료되었습니다! 엑셀(CSV) 파일로 저장 중입니다. (몇 분 정도 소요될 수 있습니다...)")
    
#     df = pd.DataFrame({
#         "text": list(generated_data),
#         "label": 1  # 1점 = 범죄 대화
#     })
    
#     output_filename = "2026_ULTIMATE_drug_slang_500k.csv"
#     df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
#     end_time = time.time()
#     print(f"\n[Success] 완벽합니다! 중복 없는 순도 100% 최신 마약 데이터 {len(df):,}건이 생성되었습니다.")
#     print(f"[Success] 총 소요 시간: {(end_time - start_time)/60:.1f}분")
#     print(f"[Success] 파일명: {os.path.abspath(output_filename)}")

# if __name__ == "__main__":
#     # 500만 건(5,000,000)으로 세팅 완료
#     generate_5M_ultimate_drug_dataset(target_count=500000)


import random
import csv
import os
import time

def generate_national_level_drug_dataset(target_count=500000):
    print(f"==================================================")
    print(f"🔥 [System] 전국구 통합 마약 범죄 문맥 생성 엔진 가동 🔥")
    print(f"==================================================")

    # 1. 전국 주요 거점 (서울, 부산, 대구, 인천 등 주요 역 대거 추가)
    stations = [
        "강남역", "홍대입구역", "서울역", "잠실역", "신림역", "건대입구역", "사당역", "고속터미널역",
        "수원역", "인천역", "부평역", "의정부역", "천안역", "대전역", "서대전역", "동성로",
        "대구역", "반월당역", "상무역", "광주역", "광주송정역", "울산역", "태화강역", "창원역",
        "마산역", "해운대역", "서면역", "남포동", "사상역", "연산역", "덕천역", "동래역",
        "청주역", "전주역", "원주역", "제주공항", "포항역", "순천역", "목포역"
    ]

    # 2. 마약 은어 (국내외 총망라)
    drug_slangs = [
        "아이스", "작대기", "찬술", "빙두", "크리스탈", "눈꽃", "연필", "피로회복제", "가루",
        "블로우", "코크", "스노우", "캔디", "몰리", "도리", "츄파춥스", "엠페러", "스컬",
        "엠제이", "그래스", "위드", "떨", "고기", "브액", "덕트", "라바", "허브",
        "켓", "케이", "원석케이", "쉬룸", "우표", "엘시드", "아파치", "하얀중국", "댄스피버"
    ]
    
    # 3. 무작위 수량 생성 (숫자 과적합 방지)
    def get_random_amount():
        units = ["g", "그램", "mg", "미리", "개", "피스", "장", "알", "입"]
        num = round(random.uniform(0.1, 9.9), 1)
        if random.random() < 0.2:
            return random.choice(["한 잔", "반 잔", "한 피스", "샘플", "시료", "소량"])
        return f"{num}{random.choice(units)}"

    # 4. 다이나믹 전국구 장소 생성기
    def get_national_location():
        # 특정 역 근처 패턴
        if random.random() < 0.6:
            station = random.choice(stations)
            exit_num = random.randint(1, 15)
            spot = random.choice(["물품보관함", "코인락커", "배전반", "실외기 뒤", "화장실 변기 뒤", "소화전"])
            num = random.randint(1, 999)
            return f"{station} {exit_num}번 출구 쪽 {spot} {num}번"
        # 일반 주거지/산지 패턴
        else:
            region = random.choice(["빌라", "아파트", "원룸", "상가", "공원", "야산", "뒷골목"])
            detail = random.choice(["단자함", "우편함", "소화기 안", "에어컨 실외기", "배수구", "환풍구", "나무 밑"])
            num1 = random.randint(101, 999)
            num2 = random.randint(1, 30)
            return f"{region} {num1}동 {num2}층 근처 {detail}"

    # 5. 유의어 믹스
    loc_synonyms = ["좌표", "위치", "포인트", "장소", "박은곳", "좌표점", "사진속거기"]
    check_synonyms = ["확인", "체크", "쳌", "컨펌", "봤냐", "확인요", "보세요"]
    actions = ["던짐", "드랍", "작업끝", "세팅완료", "붙여둠", "묻어둠", "드랍완료"]
    
    templates = [
        "입금확인. {location}에 {slang} {amount} {action}.",
        "{slang} {amount} 구함. {location} {action} 가능한가요?",
        "{location} {loc_name}. {slang} {amount} {check}요.",
        "{slang} {amount} {location} {action}. {check}바람.",
        "방금 {slang} {amount} {location} {action} 했습니다."
    ]

    output_filename = "2026_DRUG_TRAIN_DATA_NATIONAL.csv"
    
    with open(output_filename, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        
        seen = set()
        count = 0
        start_time = time.time()
        
        while count < target_count:
            sentence = random.choice(templates).format(
                location=get_national_location(),
                slang=random.choice(drug_slangs),
                amount=get_random_amount(),
                loc_name=random.choice(loc_synonyms),
                check=random.choice(check_synonyms),
                action=random.choice(actions)
            )
            
            if sentence not in seen:
                writer.writerow([sentence, 1])
                seen.add(sentence)
                count += 1
                if count % 100000 == 0:
                    print(f" -> {count:,}개 생성 완료... (전국구 좌표 조합 중)")

    print(f"\n[Success] {output_filename} 저장 완료! (총 {count:,}건)")
    print(f"전국 모든 주요 역과 은닉 장소가 완벽하게 반영되었습니다.")

if __name__ == "__main__":
    generate_national_level_drug_dataset(500000)