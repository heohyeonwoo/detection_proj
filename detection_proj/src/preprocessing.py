import pandas as pd
import numpy as np
import torch
import re
from datetime import datetime
import os
import zipfile
import glob
import json

class DataPreprocessor:
    def __init__(self, target_path="data/raw"): 
        self.target_path = target_path
        self.df = None

    def load_data(self):
        print(f"\n[System] '{self.target_path}' 경로의 데이터(zip, txt, csv, json) 스캔 시작...")
        
        if not os.path.exists(self.target_path):
            print(f"[Error] 경로를 찾을 수 없습니다: {self.target_path}")
            return

        # 1. ZIP 파일 자동 압축 해제
        if os.path.isdir(self.target_path):
            zip_files = glob.glob(os.path.join(self.target_path, '**', '*.zip'), recursive=True)
            for zip_file in zip_files:
                extract_dir = zip_file.replace('.zip', '')
                if not os.path.exists(extract_dir):
                    print(f"[Process] 압축 해제 중... -> {os.path.basename(zip_file)}")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

        # 2. 파일 스캔
        all_files = []
        if os.path.isfile(self.target_path):
            all_files.append(self.target_path)
        else:
            all_files.extend(glob.glob(os.path.join(self.target_path, '**', '*.txt'), recursive=True))
            all_files.extend(glob.glob(os.path.join(self.target_path, '**', '*.csv'), recursive=True))
            all_files.extend(glob.glob(os.path.join(self.target_path, '**', '*.json'), recursive=True))

        if not all_files:
            print("[Error] 분석할 파일이 없습니다.")
            return

        print(f"[Process] 발견된 총 {len(all_files)}개의 파일을 병합 파싱합니다...")

        parsed_data = []
        fail_count = 0
        
        # 🌟 카카오톡 3대장 정규식 (오전/오후 완벽 대응)
        txt_pattern_pc = re.compile(r'^\[(.*?)\]\ \[(.*?)\]\ (.*)$')
        txt_pattern_mobile = re.compile(r'^\d{4}년 \d{1,2}월 \d{1,2}일 .*?, (.*?) : (.*)$') 
        txt_pattern_ios = re.compile(r'^\d{4}\.\ \d{1,2}\.\ \d{1,2}\.\ (?:오전|오후)?\ ?\d{1,2}:\d{2},\ (.*?)\ :\ (.*)$') 

        for idx, file_path in enumerate(all_files):
            ext = file_path.lower().split('.')[-1]
            encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949']
            
            # ------------------------------------------------
            # 📄 TXT 파일 파싱
            # ------------------------------------------------
            if ext == 'txt':
                success = False
                for enc in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            last_sender = "Unknown"
                            
                            for line in f:
                                line = line.strip()
                                if not line: continue 
                                
                                if line.startswith("---") and line.endswith("---"): continue
                                
                                match_pc = txt_pattern_pc.match(line)
                                match_mob = txt_pattern_mobile.match(line)
                                match_ios = txt_pattern_ios.match(line)

                                sender, message = None, None

                                if match_pc:
                                    sender, message = match_pc.group(1), match_pc.group(3)
                                elif match_mob:
                                    sender, message = match_mob.group(1), match_mob.group(2)
                                elif match_ios:
                                    sender, message = match_ios.group(1), match_ios.group(2)
                                else:
                                    sender = last_sender
                                    message = line

                                if sender and message:
                                    last_sender = sender
                                    parsed_data.append({
                                        'datetime': datetime.now(), 
                                        'sender': sender,
                                        'message': message,
                                        'is_media': 1 if message.startswith("사진") or "이모티콘" in message else 0
                                    })
                        success = True
                        break
                    except Exception: continue
                if not success: fail_count += 1

            # ------------------------------------------------
            # 📄 JSON 파일 파싱
            # ------------------------------------------------
            elif ext == 'json':
                success = False
                for enc in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            json_data = json.load(f)
                            if 'sessionInfo' in json_data:
                                for session in json_data['sessionInfo']:
                                    for turn in session.get('dialog', []):
                                        msg = turn.get('utterance', '')
                                        if msg: parsed_data.append({'datetime': datetime.now(), 'sender': turn.get('speaker', 'user'), 'message': msg, 'is_media': 0})
                            elif 'data' in json_data:
                                for item in json_data['data']:
                                    msg = item.get('instruct_text', '')
                                    if msg: parsed_data.append({'datetime': datetime.now(), 'sender': item.get('publisher', 'user'), 'message': msg, 'is_media': 0})
                        success = True
                        break
                    except Exception: continue
                if not success: fail_count += 1

            # ------------------------------------------------
            # 📄 CSV/Excel 파일 파싱 (🌟 스크린샷 양식 완벽 대응 추가)
            # ------------------------------------------------
            elif ext == 'csv':
                success = False
                for enc in encodings_to_try:
                    try:
                        temp_df = pd.read_csv(file_path, encoding=enc)
                        
                        # 1. 헤더 무시하고 무조건 첫 번째(0번), 두 번째(1번) 열만 추출
                        if len(temp_df.columns) >= 2:
                            temp_df = temp_df.iloc[:, :2]
                            temp_df.columns = ['sender', 'message']
                            
                            # 2. '인물'이라는 글자가 보낸 사람 이름으로 들어갔다면 행 삭제
                            if temp_df['sender'].astype(str).str.contains('인물').any():
                                temp_df = temp_df[temp_df['sender'] != '인물']
                                
                            # 3. 빈칸(결측치) 완벽 필터링
                            temp_df = temp_df.dropna(subset=['message'])
                            temp_df = temp_df[temp_df['message'].astype(str).str.strip() != '']
                            
                            # 4. parsed_data 리스트에 통합
                            for _, row in temp_df.iterrows():
                                msg = str(row['message']).strip()
                                parsed_data.append({
                                    'datetime': datetime.now(),
                                    'sender': str(row['sender']).strip(),
                                    'message': msg,
                                    'is_media': 1 if msg.startswith("사진") or "이모티콘" in msg else 0
                                })
                        success = True
                        break
                    except Exception: continue
                if not success: fail_count += 1

        if not parsed_data:
            print("[Error] 🚨 분석 가능한 대화가 0건입니다. 파일 양식을 확인해주세요.")
            return

        # 병합 및 후처리
        self.df = pd.DataFrame(parsed_data)
        self.df['datetime'] = self.df['datetime'].fillna(datetime.now())
        
        if not self.df.empty:
            self.df['time_diff'] = 10 
            self.df['behavior_score'] = np.where(self.df['is_media'] == 1, 0.9, 0.1)

        print(f"\n[Success] 💥 파싱 대성공! 총 {len(self.df)}건의 대화가 정상적으로 로드되었습니다!")
        
        # 💡 방금 읽어들인 데이터 상위 5줄 출력 (테스트 확인용)
        pd.set_option('display.unicode.east_asian_width', True) # 한글 줄맞춤
        print("-" * 50)
        print(self.df[['sender', 'message']].head(5).to_string(index=False))
        print("-" * 50)

    def to_tensor(self):
        if getattr(self, 'df', None) is None or self.df.empty:
            print("[Warning] 변환할 데이터가 없습니다.")
            return torch.tensor([], dtype=torch.float32)
            
        tensor_data = torch.tensor(self.df['behavior_score'].values, dtype=torch.float32)
        print(f"[System] 데이터를 PyTorch 텐서로 변환 완료! (크기: {tensor_data.shape})")
        return tensor_data