import torch
import pandas as pd

class KoBERTModel:
    def __init__(self, dataframe):
        # 전처리된 엑셀 표 가져오기
        self.df = dataframe
        #
        # AI 가동 코드
        # 실제 KoBERT 가중치(.pt) 파일
        #
        
    def get_context_score(self):
        print("\n[System] KoBERT NLP 엔진이 텍스트 문맥을 분석합니다...")
        
        if self.df is None or self.df.empty:
            print("[Error] 분석할 텍스트 데이터가 없습니다.")
            return None
        
        # 대화 내용 리스트
        messages = self.df['message'].tolist()
        scores = []
        
        # 일단 걸리게 하는 키워드 ? 로 대충 일단 테스트 해보자구 
        danger_keywords = ["마약", "얼음", "작대기", "캔디", "좌표", "아이스"]
        
        for msg in messages:
            #
            # AI 추론 코드
            #
            
            # [MVP 시뮬레이션 로직]
            # 기본 문맥 위험도는 0.05정상, 의심 문맥 0.95로 일단 지정
            score = 0.05
            for keyword in danger_keywords:
                if keyword in msg:
                    score = 0.95
                    break
            scores.append(score)
        
        # 문맥 점수를 파이토치로 변환 
        context_tensor = torch.tensor(scores, dtype=torch.float32)
        print(f"[Success] 문맥 위험도 산출 완료! (텐서 크기: {context_tensor.shape})")
        
        # 위험도가 높은 대화 알려주는 법
        danger_idx = (context_tensor > 0.8).nonzero(as_tuple=True)[0]
        if len(danger_idx) > 0:
            print(f"  -> [위험감지] '{messages[danger_idx[0]]}' (위험도: {context_tensor[danger_idx[0]]:.2f})")
            
        return context_tensor