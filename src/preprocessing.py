import pandas as pd
import numpy as np
import torch
import re
from datetime import datetime
import os

class DataPreprocessor:
    def __init__(self, raw_file_path="data/raw/Talk_2026.4.3 14_57-1.txt"):  #데이터 가져오게 지정 
        self.raw_file_path = raw_file_path
        self.df = None

    def load_data(self):
        print("[System] 카카오톡 원본 데이터를 로드하고 정규화 파싱을 시작")
        
        if not os.path.exists(self.raw_file_path):
            print(f"[Error] {self.raw_file_path} 파일 ㄴ 파일을 data/raw 폴더 ㄱ")
            return

        parsed_data = []
        # 정규표현식 시간 이름 대화 내용 나누는 법
        pattern = re.compile(r'^(\d{4}\.\ \d{1,2}\.\ \d{1,2}\.\ \d{1,2}:\d{2}),\ (.*?)\ :\ (.*)$')

        #  텍스트 파일을 한 줄씩 읽으며 파싱
        with open(self.raw_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                match = pattern.match(line)
                
                if match:
                    date_time_str = match.group(1) # 시간
                    sender = match.group(2)        # 발신자
                    message = match.group(3)       # 대화 내용

                    # 시각 개체로 변경
                    dt_obj = datetime.strptime(date_time_str, '%Y. %m. %d. %H:%M')
                    
                    # 미디어 여부
                    is_media = 1 if message.startswith("사진") or message.startswith("파일:") or "이모티콘" in message else 0

                    parsed_data.append({
                        'datetime': dt_obj,
                        'sender': sender,
                        'message': message,
                        'is_media': is_media
                    })

        # 표형태로 변환
        self.df = pd.DataFrame(parsed_data)

        # 메시지 응답 시간차 계산 즉 행위 위험도 점수
        if not self.df.empty:
            self.df['time_diff'] = self.df['datetime'].diff().dt.total_seconds().fillna(0)
            
            # 일단 대충 테스트 용이니 60초 이내에 연속으로 보냈다면 위험도 0.9, 아니면 0.1 부여
            self.df['behavior_score'] = np.where((self.df['is_media'] == 1) & (self.df['time_diff'] < 60), 0.9, 0.1)

        print(f"[Success] 총 {len(self.df)}건의 대화가 정규화 파싱되어 표(DataFrame)로 만들어졌습니다.")
        print(self.df[['datetime', 'sender', 'is_media', 'behavior_score']].head())

    def to_tensor(self):
        print("\n[System] 계산된 행위 위험도를 PyTorch Tensor 자료형으로 변환합니다...")
        
        if self.df is not None and not self.df.empty:
            # 위험도 바로 나옴 행위 위험도 점수
            behavior_array = self.df['behavior_score'].to_numpy(dtype=np.float32)
            
            # GNN이 알게 PyTorch Tensor로 래핑
            tensor_data = torch.tensor(behavior_array)
            print(f"[Success] Tensor 변환 완료! (텐서 크기: {tensor_data.shape})")
            
            return tensor_data
        else:
            print("[Error] 변환할 데이터가 없습니다.")
            return None