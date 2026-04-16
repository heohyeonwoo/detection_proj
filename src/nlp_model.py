import os
import torch
import torch.nn.functional as F
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

class KoBERTModel:
    def __init__(self, dataframe):
        self.df = dataframe
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[System] 수사 엔진 기동 (Device: {self.device})")
        
        self.klue_path = "./FINAL_INVESTIGATION_MODEL"
        try:
            self.klue_tokenizer = AutoTokenizer.from_pretrained(self.klue_path)
            self.klue_model = AutoModelForSequenceClassification.from_pretrained(self.klue_path).to(self.device)
            self.klue_model.eval()
            self.klue_loaded = True
        except Exception as e:
            print(f"[Error] 최종 문맥 모델 로딩 실패: {e}")
            self.klue_loaded = False

        self.electra_path = "./koelectra_slang_hunter_v3/checkpoint-1792"
        try:
            self.electra_tokenizer = AutoTokenizer.from_pretrained(self.electra_path)
            self.electra_model = AutoModelForTokenClassification.from_pretrained(self.electra_path).to(self.device)
            self.electra_model.eval()
            self.electra_loaded = True
        except Exception as e:
            print(f"[Warning] 은어 탐지 모델 로딩 실패 (1차 문맥 모델만 가동합니다): {e}")
            self.electra_loaded = False

    def clean_message(self, raw_text):
        """
        [탐지율 복구 핵심] AI가 헷갈려하는 (날짜)와 [이름] : 껍데기를 날려버립니다.
        """
        text = str(raw_text)
        text = re.sub(r'^\*?\s*\([\d\-\s\:\.]+\)\s*', '', text)
        text = re.sub(r'\[.*?\]\s*:\s*', '', text)
        return text.strip()

    def get_context_score(self):
        if self.df is None or self.df.empty:
            return None
        
        print("[Process] 대화 문맥 및 마약 은어 전수 스캔 시작")
        messages = self.df['message'].tolist()
        scores = []
        
        if not self.klue_loaded:
            print("[Error] 핵심 엔진이 없어 분석을 중단합니다.")
            return torch.zeros(len(messages), dtype=torch.float32)

        for idx, msg in enumerate(messages):
            raw_text = str(msg).strip()
            
            text = self.clean_message(raw_text)
            
            if len(text) > 150 or text in ["이모티콘", "사진", "동영상", "음성메시지", "파일"] or len(text) < 2:
                scores.append(0.0)
                continue
                
            cleaned = re.sub(r'[^\w\sㄱ-ㅎ가-힣a-zA-Z0-9]', '', text)
            if len(cleaned.replace('ㅋ', '').replace('ㅎ', '').replace('ㅠ', '').strip()) == 0:
                scores.append(0.0)
                continue

            inputs = self.klue_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                klue_out = self.klue_model(**inputs)
            
            klue_probs = F.softmax(klue_out.logits, dim=-1)[0]
            context_score = klue_probs[1].item() 

            max_slang_prob = 0.0
            found_slangs = []

            if self.electra_loaded:
                e_inputs = self.electra_tokenizer(text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    e_out = self.electra_model(**e_inputs)
                
                e_probs = F.softmax(e_out.logits, dim=-1)[0]
                e_preds = torch.argmax(e_out.logits, dim=-1)[0]
                tokens = self.electra_tokenizer.convert_ids_to_tokens(e_inputs["input_ids"][0])
                
                for token, pred, prob in zip(tokens, e_preds, e_probs):
                    if token in self.electra_tokenizer.all_special_tokens:
                        continue
                    
                    slang_prob = prob[1].item()
                    if pred.item() == 1 or slang_prob > 0.6: 
                        clean_token = token.replace("##", "")
                        found_slangs.append(f"[{clean_token}: {slang_prob*100:.1f}%]")
                        max_slang_prob = max(max_slang_prob, slang_prob)
            
            danger_score = max(context_score, max_slang_prob)

            if danger_score > 0.5:
                print(f"[분석결과] {idx+1}행: '{raw_text[:30]}...'")
                print(f" -> 위험도: {danger_score*100:.1f}% (문맥: {context_score*100:.1f}% | 은어: {', '.join(found_slangs) if found_slangs else '없음'})")
            
            scores.append(danger_score)
        
        context_tensor = torch.tensor(scores, dtype=torch.float32)
        print(f"[Success] 분석 완료 (총 {len(context_tensor)}건)")
        
        return context_tensor