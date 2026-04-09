import torch
import numpy as np
import re
from transformers import BertModel, BertTokenizer
from sklearn.ensemble import IsolationForest

class KoBERTModel:
    def __init__(self, df):
        self.df = df
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System] NLP 연산 장치: {self.device}")
        
        MODEL_PATH = "./my_trained_model"
        
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertModel.from_pretrained(MODEL_PATH)
        self.model.to(self.device)
        self.model.eval() 
        
        self.iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    def _get_embeddings(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output

    def get_context_score(self, messages=None, batch_size=64):
        if messages is None:
            messages = self.df['message'].tolist()
            
        total_msgs = len(messages)
        if total_msgs == 0:
            return torch.tensor([])

        candidate_msgs = []
        for msg in messages:
            text = str(msg).strip()
            
            # 100자 초과 긴 글 제외
            if len(text) > 100:
                continue
            
            if text in ["이모티콘", "사진", "동영상", "음성메시지", "파일"]:
                continue
                
            cleaned = re.sub(r'[^\w\sㄱ-ㅎ가-힣a-zA-Z0-9]', '', text)
            if len(cleaned.replace('ㅋ', '').replace('ㅎ', '').replace('ㅠ', '').strip()) == 0:
                continue
                
            candidate_msgs.append(text)

        # 중복 제거
        unique_msgs = list(set(candidate_msgs))
        unique_msgs.sort(key=len)
        print(f"[Process] 전체 {total_msgs}개 중, 긴 글 및 이모티콘을 제외한 {len(unique_msgs)}개 핵심 문장만 정밀 스캔합니다.")
        
        unique_scores = {}
        if len(unique_msgs) > 0:
            unique_embeddings = {}
            self.model.eval()
            with torch.no_grad():
                for idx, i in enumerate(range(0, len(unique_msgs), batch_size)):
                    batch_msgs = unique_msgs[i:i+batch_size]
                    batch_emb = self._get_embeddings(batch_msgs).cpu().numpy()
                    
                    for msg, emb in zip(batch_msgs, batch_emb):
                        unique_embeddings[msg] = emb

            candidate_embeddings = np.array([unique_embeddings[msg] for msg in unique_msgs])

            self.iso_forest.fit(candidate_embeddings)
            scores = -self.iso_forest.score_samples(candidate_embeddings)
            
            # 0 ~ 1.0 점수 정규화
            min_score, max_score = scores.min(), scores.max()
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)
                
            for msg, score in zip(unique_msgs, normalized_scores):
                unique_scores[msg] = score

        final_scores = []
        for msg in messages:
            msg_str = str(msg).strip()
            if msg_str in unique_scores:
                final_scores.append(unique_scores[msg_str])
            else:
                final_scores.append(0.0) # 긴 글, 이모티콘은 무조건 위험도 0점 처리

        return torch.tensor(final_scores, dtype=torch.float32)