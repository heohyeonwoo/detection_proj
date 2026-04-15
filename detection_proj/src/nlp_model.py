import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class KoBERTModel:
    def __init__(self, dataframe=None):
        self.df = dataframe
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"\n[System]  AI: 타임라인 추적 & 수사 패턴 추출 ... (장치: {self.device})")
        
        self.klue_path = "final_jun_context_model" 
        try:
            self.klue_tokenizer = AutoTokenizer.from_pretrained(self.klue_path)
            self.klue_model = AutoModelForSequenceClassification.from_pretrained(
                self.klue_path, num_labels=2
            ).to(self.device)
            self.klue_model.eval()
            print("[Success] AI 문맥 분석 모듈 로딩 완료 (외부 모델/금지어 제거됨)")
        except Exception as e:
            print(f"[Error] 모델 로딩 실패: {e}")
            self.klue_model = None

    def extract_drug_slang(self, text_block):
        """
        수사관 전용 팩트 추출기: 범죄 타임라인 내부에서 명확한 거래 패턴만 스캔하여 딱 한 단어만 추출
        """
        # 패턴 1: [단어] + [숫자] + [거래 단위(g, 장, 개, ml 등)]
        pattern1 = r'([가-힣a-zA-Z]+)\s*(?:[0-9]+(?:g|그램|장|개|앰플|ml|cc|정))'
        matches = re.findall(pattern1, text_block)
        
        if matches:
            # 수량 앞에 붙는 흔한 부사 예외 처리
            valid_matches = [m for m in matches if m not in ['오늘', '총', '약', '딱']]
            if valid_matches:
                return valid_matches[0] # 가장 먼저 발견된 확실한 대상 딱 1개만 리턴
        
        # 패턴 2: 단위를 안 썼을 경우 '원합니다, 주세요, 있나요' 앞의 단어
        pattern2 = r'([가-힣a-zA-Z]+)\s*(?:구합니다|원합니다|있나요|주세요)'
        matches2 = re.findall(pattern2, text_block)
        if matches2:
            valid_matches2 = [m for m in matches2 if len(m) >= 2]
            if valid_matches2:
                return valid_matches2[0]

        return "탐지 불가 (정황만 존재)"

    def extract_drug_slang(self, text_block):
        """
        수사관 전용 팩트 추출기: 범죄 타임라인 내부에서 명확한 거래 패턴만 스캔하여 딱 한 단어만 추출
        """
        # 패턴 1: [단어] + [숫자] + [거래 단위(g, 장, 개, ml 등)]
        pattern1 = r'([가-힣a-zA-Z]+)\s*[0-9]+(?:g|그램|장|개|앰플|ml|cc|정)'
        matches = re.findall(pattern1, text_block)
        
        if matches:
            valid_matches = [m for m in matches if m not in ['오늘', '총', '약', '딱']]
            if valid_matches:
                return valid_matches[0] 
        
        # 패턴 2: 단위를 안 썼을 경우 '원합니다, 주세요, 있나요' 앞의 단어
        pattern2 = r'([가-힣a-zA-Z]+)\s*(?:구합니다|원합니다|있나요|주세요)'
        matches2 = re.findall(pattern2, text_block)
        if matches2:
            # 혹시 모를 일상어 예외 처리
            valid_matches2 = [m for m in matches2 if len(m) >= 2 and m not in ['거래', '사진', '종류', '무게', '시간', '가격', '연락']]
            if valid_matches2:
                return valid_matches2[0]

        return "탐지 불가 (정황만 존재)"

    def get_context_score(self, messages=None, **kwargs):
        print("\n[System] 수사 시작: 범죄 정황을 하나의 '타임라인'으로 묶습니다.")
        
        if messages is None:
            messages = self.df['message'].tolist() if self.df is not None else []
        
        n = len(messages)
        if n == 0 or self.klue_model is None: 
            return torch.tensor([]), torch.tensor([])

        raw_scores = []
        for msg in messages:
            msg_str = str(msg).strip()
            if not msg_str: 
                raw_scores.append(0.0)
                continue
            inputs = self.klue_tokenizer(msg_str, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.klue_model(**inputs)
            raw_scores.append(F.softmax(outputs.logits, dim=-1)[0][1].item())

        # 🌟 핵심 수정 포인트: 간격을 5줄 -> 15줄로 대폭 늘림 (중간에 단서가 안 빠져나가게!)
        is_crime = [score > 0.30 for score in raw_scores]
        crime_indices = [i for i, val in enumerate(is_crime) if val]

        if crime_indices:
            for i in range(len(crime_indices) - 1):
                if crime_indices[i+1] - crime_indices[i] <= 15: # <- 이 부분입니다!
                    for j in range(crime_indices[i] + 1, crime_indices[i+1]):
                        is_crime[j] = True

        conspiracy_scores = np.array(raw_scores)
        slang_scores = np.zeros(n)
        
        blocks = []
        current_block = None

        for idx, flag in enumerate(is_crime):
            if flag:
                if current_block is None:
                    current_block = {'start': idx, 'end': idx, 'msgs': [], 'raw_texts': []}
                current_block['end'] = idx
                current_block['msgs'].append(f"[{idx+1}] {self.df['sender'].iloc[idx] if self.df is not None else '사용자'}: {messages[idx]}")
                current_block['raw_texts'].append(str(messages[idx]))
                slang_scores[idx] = 0.95
            else:
                if current_block is not None:
                    blocks.append(current_block)
                    current_block = None
        if current_block is not None:
            blocks.append(current_block)

        if not blocks:
            print("\n[System]  탐지된 범죄 모의 구간이 없습니다. (모두 일상 대화)")
        else:
            for b_idx, block in enumerate(blocks):
                full_text = " ".join(block['raw_texts'])
                
                # 🌟 거대한 타임라인 안에서 알맹이만 쏙 빼먹기
                best_slang = self.extract_drug_slang(full_text)
                
                print("\n" + "="*60)
                print(f"[범죄 타임라인 #{b_idx+1}] 라인 {block['start']+1} ~ {block['end']+1} 구간")
                print("-" * 60)
                for m in block['msgs']:
                    print(f"   {m}")
                print("-" * 60)
                print(f"    거래 타임라인의 은어: {best_slang}")
                print("="*60)

        print(f"\n[Success] 타임라인 병합 및 팩트 추출 완료")
        return torch.tensor(conspiracy_scores, dtype=torch.float32), torch.tensor(slang_scores, dtype=torch.float32)