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
        print(f"\n[System] '{self.target_path}' 경로의 모든 데이터(zip, txt, csv, json) 스캔 시작...")
        
        if not os.path.exists(self.target_path):
            print(f"[Error] 경로를 찾을 수 없습니다: {self.target_path}")
            return

        # 하위 폴더까지 싹 다 뒤져서 압축 파일 자동 해제
        if os.path.isdir(self.target_path):
            zip_files = glob.glob(os.path.join(self.target_path, '**', '*.zip'), recursive=True)
            for zip_file in zip_files:
                extract_dir = zip_file.replace('.zip', '')
                if not os.path.exists(extract_dir):
                    print(f"[Process] 압축 해제 중... -> {os.path.basename(zip_file)}")
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

        # 모든 지원 파일 찾기 (txt, csv, json)
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

        print(f"[Process] 발견된 총 {len(all_files)}개의 파일을 포맷에 맞게 병합 파싱합니다. (시간이 다소 소요될 수 있습니다...)")

        parsed_data = []
        txt_pattern = re.compile(r'^(\d{4}\.\ \d{1,2}\.\ \d{1,2}\.\ \d{1,2}:\d{2}),\ (.*?)\ :\ (.*)$')
        
        fail_count = 0 # 에러 메시지 도배 방지용 카운터

        # 파일 확장자에 따라 알아서 파싱 방법 변경
        for idx, file_path in enumerate(all_files):
            # 현재 진행률
            if (idx + 1) % 1000 == 0:
                print(f"  -> 데이터 병합 중 ({idx + 1} / {len(all_files)})")
            
            ext = file_path.lower().split('.')[-1]
            encodings_to_try = ['utf-8', 'utf-8-sig', 'cp949']
            
            # 카카오톡 TXT 형식 처리
            if ext == 'txt':
                success = False
                for enc in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            for line in f:
                                match = txt_pattern.match(line.strip())
                                if match:
                                    try: dt_obj = datetime.strptime(match.group(1), '%Y. %m. %d. %H:%M')
                                    except: continue
                                    message = match.group(3)
                                    parsed_data.append({
                                        'datetime': dt_obj,
                                        'sender': match.group(2),
                                        'message': message,
                                        'is_media': 1 if message.startswith("사진") or message.startswith("파일:") or "이모티콘" in message else 0
                                    })
                        success = True
                        break
                    except Exception: continue
                if not success: fail_count += 1

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

            # JSON 형식 처리 
            elif ext == 'json':
                success = False
                for enc in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            json_data = json.load(f)
                            
                            if isinstance(json_data, dict):
                                documents = json_data.get('document', json_data.get('data', []))
                                if not isinstance(documents, list): documents = [documents]
                                for doc in documents:
                                    utterances = doc.get('utterance', doc.get('messages', []))
                                    for utt in utterances:
                                        msg = utt.get('form', utt.get('text', utt.get('message', '')))
                                        sndr = utt.get('speaker_id', utt.get('sender', 'Unknown'))
                                        dt_str = doc.get('created_at', str(datetime.now())) 
                                        if msg:
                                            parsed_data.append({'datetime': pd.to_datetime(dt_str, errors='coerce') or datetime.now(), 'sender': str(sndr), 'message': str(msg), 'is_media': 0})
                            
                            elif isinstance(json_data, list):
                                for item in json_data:
                                    msg = item.get('text', item.get('message', item.get('utterance', '')))
                                    sndr = item.get('sender', item.get('user_id', item.get('author', 'Unknown')))
                                    dt_str = item.get('date', item.get('created_at', str(datetime.now())))
                                    if msg:
                                        parsed_data.append({'datetime': pd.to_datetime(dt_str, errors='coerce') or datetime.now(), 'sender': str(sndr), 'message': str(msg), 'is_media': 0})
                        success = True
                        break
                    except Exception: continue
                if not success: fail_count += 1

        if fail_count > 0:
            print(f" - [참고] 형식이 맞지 않거나 손상된 파일 {fail_count}개는 안전하게 건너뛰었습니다.")

        if not parsed_data:
            print("[Error] 분석 가능한 대화 데이터가 한 건도 없습니다.")
            return

        # 모든 데이터를 하나ㄹ로병합
        self.df = pd.DataFrame(parsed_data)
        self.df['datetime'] = self.df['datetime'].fillna(datetime.now())
        self.df.sort_values(by='datetime', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        if not self.df.empty:
            self.df['time_diff'] = self.df['datetime'].diff().dt.total_seconds().fillna(0)
            self.df['behavior_score'] = np.where((self.df['is_media'] == 1) & (self.df['time_diff'] < 60), 0.9, 0.1)

        print(f"[Success] 다중 포맷 자동 병합 완료! 총 {len(self.df)}건의 대화가 메모리에 로드되었습니다.")

    def to_tensor(self):
        if self.df is not None and not self.df.empty:
            behavior_array = self.df['behavior_score'].to_numpy(dtype=np.float32)
            tensor_data = torch.tensor(behavior_array)
            return tensor_data
        return None