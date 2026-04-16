import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. 경로 설정 (가장 최근의 체크포인트 폴더 경로)
PRETRAINED_MODEL_PATH = './klue_bert_local_results/checkpoint-139504' 
FOLDER_PATH = '/Users/jeongjun-yeong/Downloads/141.한국어 멀티세션 대화/01-1.정식개방데이터/Training/02.라벨링데이터'

def get_time_tag(timestamp_str):
    """시간 정보를 [새벽], [오후] 등 태그로 변환"""
    try:
        # JSON 시간 형식이 "HH:mm:ss" 또는 "HH:mm"인 경우 처리
        hour = int(timestamp_str.split(':')[0])
        if 0 <= hour < 6: return "[새벽]"
        elif 6 <= hour < 12: return "[오전]"
        elif 12 <= hour < 18: return "[오후]"
        else: return "[저녁]"
    except:
        return "[시간미상]"

def process_with_tags(file_path):
    local_records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for session in data.get('sessionInfo', []):
            for turn in session.get('dialog', []):
                text = turn.get('utterance', '')
                if not text: continue

                # 🌟 상황 태그 추가
                time_info = turn.get('time', '00:00') 
                time_tag = get_time_tag(time_info)
                
                # 사진 전송 여부 (JSON에 키가 없으면 텍스트에서 [사진] 문구 탐색)
                img_tag = "[IMG]" if "[사진]" in text or turn.get('has_photo') else ""

                full_text = f"{time_tag} {img_tag} {text}".strip()
                local_records.append({'text': full_text, 'label': 0}) # 현재는 모두 정상 라벨
    except:
        pass
    return local_records

class FinalContextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.text[index])
        label = self.data.label[index]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 최종 상황 인식 학습 시작 (장치: {device})")

    # 2. 데이터 추출 (병렬 처리 로직 포함)
    file_list = []
    for root, _, files in os.walk(FOLDER_PATH):
        for file in files:
            if file.endswith('.json'):
                file_list.append(os.path.join(root, file))
                
    print(f"총 {len(file_list)}개 파일에서 상황 태그 포함 데이터 추출 중...")
    records = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_with_tags, f) for f in file_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            records.extend(future.result())

    df = pd.DataFrame(records)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"데이터 추출 완료! 총 {len(df)}개 문장")

    # 3. 기존 학습된 모델 로드
    print(f"\n기존 모델({PRETRAINED_MODEL_PATH}) 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=2)
    model.to(device)

    # 4. 딱 1 에포크 학습 설정
    training_args = TrainingArguments(
        output_dir='./final_context_model',
        num_train_epochs=1,           # 🌟 딱 한 바퀴만 추가 공부!
        per_device_train_batch_size=32,
        save_strategy="epoch",
        logging_steps=100,
        dataloader_pin_memory=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=FinalContextDataset(train_df, tokenizer),
        eval_dataset=FinalContextDataset(val_df, tokenizer)
    )

    print("\n본격적인 최종 상황 인식 학습을 시작합니다!")
    trainer.train()