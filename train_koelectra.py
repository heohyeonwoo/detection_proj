import os
import json
import random
import pandas as pd
import torch
from torch import nn  # 🌟 손실 함수(Loss) 조작을 위해 추가
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
NORMAL_DATA_PATH = '/Users/jeongjun-yeong/Downloads/141.한국어 멀티세션 대화/01-1.정식개방데이터/Training/02.라벨링데이터'
HARMFUL_DATA_PATH = '/Users/jeongjun-yeong/Downloads/119.국가기록물 대상 초거대AI 학습을 위한 말뭉치 데이터(압축x)/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_2. 유해질의 데이터_범죄'

# ==========================================
# 2. 거대한 일상 단어 풀 만들기
# ==========================================
def build_massive_word_pool(base_path, max_words=50000):
    word_pool = set()
    print("1단계: 정상 대화에서 수만 개의 '가짜 은어' 사전 수집 중.")
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json') and len(word_pool) < max_words:
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for session in data.get('sessionInfo', []):
                        for turn in session.get('dialog', []):
                            text = turn.get('utterance', '')
                            words = [w for w in text.split() if len(w) >= 2]
                            word_pool.update(words)
                except:
                    pass
            if len(word_pool) >= max_words: break
                
    pool_list = list(word_pool)
    print(f"✅ 수집 완료! 총 {len(pool_list)}개의 방대한 가짜 은어 확보!")
    return pool_list

# ==========================================
# 3. 데이터 뻥튀기(증강) 및 추출 로직
# ==========================================
def load_augmented_harmful_data(base_path, word_pool, augment_factor=10):
    records = []
    print(f"\n🕵️‍♂️ 2단계: 범죄 데이터 읽기 시작! (경로: {base_path})")
    
    file_count = 0
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                file_count += 1
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                    
                    for item in data.get('data', []):
                        text = item.get('instruct_text', '')
                        words = text.split()
                        
                        if len(words) > 3: 
                            for _ in range(augment_factor):
                                augmented_words = words.copy()
                                labels = [0] * len(augmented_words)
                                
                                target_idx = random.randint(0, len(augmented_words) - 1)
                                fake_word = random.choice(word_pool)
                                
                                augmented_words[target_idx] = fake_word
                                labels[target_idx] = 1 
                                
                                records.append({'words': augmented_words, 'labels': labels})
                
                except Exception as e:
                    print(f"❌ [에러 발생 파일] {file} | 사유: {e}")

    print(f"📂 발견된 JSON 파일 총 개수: {file_count}개")
    print(f"✅ 데이터 뻥튀기 성공! 총 {len(records)}개의 범죄 치환 데이터 생성 완료!")
    return records

def load_normal_data(base_path, target_count):
    records = []
    print(f"\n⚖️ 3단계: 황금 밸런스를 위해 정상 대화를 딱 {target_count}개만 추출합니다...")
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json') and len(records) < target_count:
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for session in data.get('sessionInfo', []):
                        for turn in session.get('dialog', []):
                            if len(records) >= target_count: break
                            
                            text = turn.get('utterance', '')
                            words = text.split()
                            if len(words) > 3:
                                labels = [0] * len(words) 
                                records.append({'words': words, 'labels': labels})
                except:
                    pass
            if len(records) >= target_count: break
                
    print(f"✅ 일상 데이터 {len(records)}개 추출 완료!")
    return records

# ==========================================
# 4. KoELECTRA 전용 데이터셋 클래스
# ==========================================
class SmartSlangDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        tokenized_inputs = self.tokenizer(
            record['words'], is_split_into_words=True,
            max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        labels = []
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            else:
                labels.append(record['labels'][word_idx])
                
        return {
            'input_ids': tokenized_inputs['input_ids'].flatten(),
            'attention_mask': tokenized_inputs['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# ==========================================
# 🚨 5. 특단 조치: AI의 꼼수를 막는 가중치 트레이너
# ==========================================
class SlangWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 은어(1)를 틀리면 정상(0)을 틀렸을 때보다 15배 더 큰 페널티를 부여!
        weight = torch.tensor([1.0, 15.0]).to(model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        loss = loss_fct(active_logits, active_labels)
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 6. 메인 실행
# ==========================================
if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nAI 엔진 가동 준비 완료! (장치: {device})")

    massive_word_pool = build_massive_word_pool(NORMAL_DATA_PATH)
    crime_records = load_augmented_harmful_data(HARMFUL_DATA_PATH, massive_word_pool, augment_factor=10)
    normal_records = load_normal_data(NORMAL_DATA_PATH, target_count=len(crime_records))
    
    all_records = crime_records + normal_records
    random.shuffle(all_records)
    print(f"\n1:1 비율 데이터셋 완성 (총 {len(all_records)} 문장)")

    split_idx = int(len(all_records) * 0.8)
    train_data = all_records[:split_idx]
    val_data = all_records[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_dataset = SmartSlangDataset(train_data, tokenizer)
    val_dataset = SmartSlangDataset(val_data, tokenizer)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 🌟 저장 폴더를 v3로 변경
    training_args = TrainingArguments(
        output_dir='./koelectra_slang_hunter_v3',
        num_train_epochs=2,           
        per_device_train_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        dataloader_pin_memory=False,
        report_to="none"
    )

    # 🌟 일반 Trainer 대신 가중치 트레이너(SlangWeightedTrainer) 사용!
    trainer = SlangWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    print("\n⚔️ 꼼수 금지! 가중치 회초리를 적용한 진짜 문맥 학습을 시작합니다!")
    trainer.train()