import os
import glob
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. 경로 및 설정
JSON_DIR = "/Users/jeongjun-yeong/Downloads/141.한국어 멀티세션 대화/01-1.정식개방데이터/Training/02.라벨링데이터/"
CRIME_CSV = "detection_proj/src/2026_DRUG_TRAIN_DATA_NATIONAL.csv"
OUTPUT_DIR = "./final_jun_context_model"
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def train_new_ai():
    # ==========================================
    # [Step 1] 정상 대화(utterance) 추출
    # ==========================================
    print(f"\n[1/3] JSON 폴더에서 진짜 대화(utterance) 추출 중...")
    json_files = glob.glob(os.path.join(JSON_DIR, "**", "*.json"), recursive=True)
    normal_texts = []
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'sessionInfo' in data:
                    for session in data['sessionInfo']:
                        for turn in session.get('dialog', []):
                            msg = turn.get('utterance', '').strip()
                            if msg: normal_texts.append(msg)
        except: continue
    
    df_normal = pd.DataFrame({'text': normal_texts, 'label': 0})
    print(f" -> 정상 대화 추출 완료: {len(df_normal)}건")

    # ==========================================
    # [Step 2] 범죄 데이터 로드 및 1:1 비율 조정
    # ==========================================
    print(f"\n[2/3] 범죄 데이터 로드 및 1:1 비율 조정 중...")
    df_crime = pd.read_csv(CRIME_CSV)
    if 'message' in df_crime.columns:
        df_crime = df_crime.rename(columns={'message': 'text'})
    df_crime['label'] = 1
    df_crime = df_crime[['text', 'label']]
    
    min_count = min(len(df_normal), len(df_crime))
    df_normal = df_normal.sample(n=min_count, random_state=42)
    df_crime = df_crime.sample(n=min_count, random_state=42)
    
    train_df = pd.concat([df_normal, df_crime]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f" -> 최종 데이터 규모: 총 {len(train_df)}건 (1:1 비율)")

    # ==========================================
    # [Step 3] 새로운 AI 학습 및 "한 번에" 저장
    # ==========================================
    print(f"\n[3/3] 새로운 AI(KLUE-BERT) 학습 시작... (장치: {DEVICE})")
    
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    
    dataset = Dataset.from_pandas(train_df)
    tokenized_dataset = dataset.map(tokenize, batched=True)

    # 🌟 수정된 TrainingArguments
    training_args = TrainingArguments(
        output_dir="./tmp_checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        save_strategy="no",      # 중간 저장 안 함!
        eval_strategy="no",      # 🌟 'evaluation_strategy'에서 'eval_strategy'로 수정!
        logging_steps=100,
        report_to="none"         # 불필요한 로그 발송 방지
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 학습 가동
    trainer.train()

    # 🌟 최종 저장 (3단계가 끝나면 딱 한 번만 저장됩니다)
    print(f"\n[Success] 학습 완료! 최종 모델을 '{OUTPUT_DIR}'에 저장합니다.")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train_new_ai()