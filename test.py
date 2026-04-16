import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 1. 학습된 모델 경로 설정
# 주의: 좌측 탐색기에서 koelectra_slang_hunter_v2 폴더 안의 '가장 숫자가 큰 checkpoint 폴더' 경로를 넣어주세요!
# 예시: './koelectra_slang_hunter_v2/checkpoint-1500'
MODEL_PATH = 'koelectra_slang_hunter_v3/checkpoint-1792' 

def load_hunter_ai():
    print(f"🚀 실전 탐지 AI 로딩 중... (경로: {MODEL_PATH})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        model.eval() # 평가 모드로 전환
        print("✅ 탐지기 가동 준비 완료!\n")
        return tokenizer, model, device
    except Exception as e:
        print(f"❌ 모델 로딩 실패! 경로(checkpoint)를 다시 확인해주세요. 에러: {e}")
        return None, None, None

def detect_slang(text, tokenizer, model, device):
    """문장을 입력받아 은어를 찾아내는 핵심 로직"""
    # 1. 문장 토크나이징 (AI가 읽을 수 있게 쪼개기)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # 2. AI의 예측 (추론)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. 확률 계산 (은어일 확률)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)[0] # 0: 정상 확률, 1: 은어 확률
    preds = torch.argmax(logits, dim=-1)[0]
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    print("-" * 50)
    print(f"🗣️ 분석 문장: '{text}'")
    print("-" * 50)
    
    found_slang = False
    
    # 4. 결과 출력
    for token, pred, prob in zip(tokens, preds, probs):
        # 특수 기호([CLS], [SEP] 등)는 무시
        if token in tokenizer.all_special_tokens: 
            continue
            
        slang_probability = prob[1].item() * 100 # 은어일 확률(%)
        
        # 원래 단어 모양으로 복원 (## 제거)
        clean_token = token.replace("##", "")
        
        # 판단 기준: AI가 은어(1)로 예측했거나, 확률이 50% 이상인 경우
        if pred.item() == 1 or slang_probability > 50:
            found_slang = True
            print(f"🚨 [은어 의심] '{clean_token}' ➔ 위험도: {slang_probability:.1f}%")
        else:
            # 정상 단어는 회색빛으로 잔잔하게 출력 (확률도 같이 보여줌)
            print(f"✅ [정상 단어] '{clean_token}' ➔ 위험도: {slang_probability:.1f}%")
            
    if not found_slang:
        print("\n🟢 특이사항 없음. 정상적인 대화 흐름입니다.")
    else:
        print("\n🔴 경고! 비정상적인 문맥(거래 정황) 속에서 치환된 단어가 발견되었습니다.")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    tokenizer, model, device = load_hunter_ai()
    
    if model:
        # ==========================================
        # 🧪 테스트해볼 문장들을 여기에 자유롭게 넣어보세요!
        # ==========================================
        test_sentences = [
            # 1. 완전히 정상적인 일상 문맥
            "오늘 저녁에 피자 먹고 컴퓨터로 게임이나 하자.",
            
            # 2. 범죄 문맥에 평범한 단어를 섞은 경우 (AI가 문맥을 파악하는지 테스트)
            "[새벽] [IMG] 피자 0.5개 던져놨어. 입금 확인되면 좌표 줄게.",
            
            # 3. 또 다른 평범한 단어를 억지로 끼워 넣은 거래 상황
            "내일 컴퓨터 한 장 시원하게 가능? 텔레로 연락해.",
            
            # 4. 일상 문맥에 '아이스'라는 단어가 들어간 경우
            "날씨가 너무 더워서 아이스 아메리카노 마시고 싶다."
        ]
        
        for sentence in test_sentences:
            detect_slang(sentence, tokenizer, model, device)