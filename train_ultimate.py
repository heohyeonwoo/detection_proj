# import os
# import glob
# import json
# import pandas as pd
# import torch
# from tqdm import tqdm
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForSequenceClassification, 
#     Trainer, 
#     TrainingArguments,
#     EarlyStoppingCallback
# )
# from sklearn.metrics import f1_score, accuracy_score

# def collect_data(base_path):
#     print(f"[Step 1] 스캔 경로: {base_path}")
    
#     # 02.라벨링데이터 폴더 내 모든 파일 탐색
#     target_pattern = os.path.join(base_path, "**/02.라벨링데이터/**/*.*")
#     all_files = glob.glob(target_pattern, recursive=True)
#     normal_texts = set()

#     if not all_files:
#         print("[Error] 파일을 찾지 못했습니다. 경로를 확인하십시오.")
#         return pd.DataFrame()

#     for f in tqdm(all_files, desc="데이터 추출"):
#         if os.path.isdir(f) or f.endswith('.zip'): continue
#         ext = os.path.splitext(f)[1].lower()
#         try:
#             if ext == '.json':
#                 with open(f, 'r', encoding='utf-8') as file:
#                     data = json.load(file)
#                     # JSON 내부의 모든 텍스트 필드(utterance, text 등) 검색
#                     def find_text(obj):
#                         if isinstance(obj, dict):
#                             for k, v in obj.items():
#                                 if k in ['utterance', 'text', 'sentence', 'form', 'content']:
#                                     if isinstance(v, str) and len(v.strip()) > 1:
#                                         normal_texts.add(v.strip())
#                                 else:
#                                     find_text(v)
#                         elif isinstance(obj, list):
#                             for item in obj:
#                                 find_text(item)
#                     find_text(data)
#             elif ext == '.csv':
#                 df = pd.read_csv(f)
#                 col = 'text' if 'text' in df.columns else df.columns[0]
#                 normal_texts.update(df[col].dropna().astype(str).tolist())
#             elif ext == '.txt':
#                 with open(f, 'r', encoding='utf-8') as file:
#                     normal_texts.update([l.strip() for l in file.readlines() if len(l.strip()) > 1])
#         except: continue
            
#     return pd.DataFrame({'text': list(normal_texts), 'label': 0})

# def run_training():
#     source_path = r"C:\Users\82109\Downloads\141.한국어 멀티세션 대화\01-1.정식개방데이터"
#     drug_path = "2026_DRUG_TRAIN_DATA_NATIONAL.csv"

#     # 정상 데이터 로드
#     df_normal = collect_data(source_path)
#     if df_normal.empty: return

#     # 악성 데이터 로드
#     df_drug = pd.read_csv(drug_path)
#     df_drug['label'] = 1

#     # 1:1 데이터 비율 조정
#     min_size = min(len(df_normal), len(df_drug))
#     df_final = pd.concat([
#         df_normal.sample(n=min_size, random_state=42),
#         df_drug.sample(n=min_size, random_state=42)
#     ]).sample(frac=1, random_state=42).reset_index(drop=True)

#     print(f"[Step 2] 데이터 통합: {len(df_final):,}건 (정상 {min_size} : 악성 {min_size})")

#     # 모델 초기화
#     model_name = "klue/bert-base"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     def tokenize_fn(examples):
#         return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

#     dataset = Dataset.from_pandas(df_final).train_test_split(test_size=0.1)
#     tokenized_ds = dataset.map(tokenize_fn, batched=True, desc="토큰화")

#     # 학습 설정 (최신 라이브러리 규격 eval_strategy 적용)
#     training_args = TrainingArguments(
#         output_dir="./DRUG_DETECTION_LOGS",
#         eval_strategy="steps",
#         eval_steps=500,
#         save_strategy="steps",
#         save_steps=500,
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=64,
#         num_train_epochs=5,
#         weight_decay=0.01,
#         load_best_model_at_end=True,
#         metric_for_best_model="f1",
#         fp16=torch.cuda.is_available(),
#         logging_steps=100,
#         report_to="none"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_ds["train"],
#         eval_dataset=tokenized_ds["test"],
#         compute_metrics=lambda p: {
#             'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1)),
#             'f1': f1_score(p.label_ids, p.predictions.argmax(-1))
#         },
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
#     )

#     print(f"[Step 3] 학습 시작 (장치: {device})")
#     trainer.train()

#     # 결과 저장
#     final_path = "./FINAL_INVESTIGATION_MODEL"
#     trainer.save_model(final_path)
#     tokenizer.save_pretrained(final_path)
#     print(f"[Step 4] 완료: {os.path.abspath(final_path)}")

# if __name__ == "__main__":
#     run_training()

# import os
# import glob
# import json
# import pandas as pd
# import torch
# from tqdm import tqdm
# from datasets import Dataset
# from sklearn.metrics import f1_score, accuracy_score
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForSequenceClassification, 
#     Trainer, 
#     TrainingArguments,
#     EarlyStoppingCallback
# )

# # [가장 중요] 에러를 뿜는 Intel GPU를 완전히 배제하고 안정적인 CPU로 강제 고정
# DEVICE = torch.device("cpu")

# def collect_data(base_path):
#     print(f"[Step 1] 경로 스캔: {base_path}")
#     target_pattern = os.path.join(base_path, "**/02.라벨링데이터/**/*.*")
#     all_files = glob.glob(target_pattern, recursive=True)
#     normal_texts = set()

#     for f in tqdm(all_files, desc="데이터 수집"):
#         if os.path.isdir(f) or f.endswith('.zip'): continue
#         ext = os.path.splitext(f)[1].lower()
#         try:
#             if ext == '.json':
#                 with open(f, 'r', encoding='utf-8') as file:
#                     data = json.load(file)
#                     def find_text(obj):
#                         if isinstance(obj, dict):
#                             for k, v in obj.items():
#                                 if k in ['utterance', 'text', 'sentence', 'form', 'content']:
#                                     if isinstance(v, str) and len(v.strip()) > 1:
#                                         normal_texts.add(v.strip())
#                                 else: find_text(v)
#                         elif isinstance(obj, list):
#                             for item in obj: find_text(item)
#                     find_text(data)
#             elif ext == '.csv':
#                 df = pd.read_csv(f)
#                 col = 'text' if 'text' in df.columns else df.columns[0]
#                 normal_texts.update(df[col].dropna().astype(str).tolist())
#         except: continue
#     return pd.DataFrame({'text': list(normal_texts), 'label': 0})

# def run_training():
#     source_path = r"C:\Users\82109\Downloads\141.한국어 멀티세션 대화\01-1.정식개방데이터"
#     drug_path = "2026_DRUG_TRAIN_DATA_NATIONAL.csv"
#     cache_path = "normal_data_cache.csv"

#     # 데이터 로드 (캐시 파일이 있으면 1초 만에 불러옵니다)
#     if os.path.exists(cache_path):
#         print(f"[Step 1] 캐시 로드: {cache_path}")
#         df_normal = pd.read_csv(cache_path)
#     else:
#         df_normal = collect_data(source_path)
#         df_normal.to_csv(cache_path, index=False, encoding='utf-8-sig')

#     df_drug = pd.read_csv(drug_path)
#     df_drug['label'] = 1

#     # 노트북이 현실적으로 완주할 수 있는 분량(정상 2만, 마약 2만 = 총 4만 건)으로 밸런싱
#     min_size = min(len(df_normal), len(df_drug), 20000)
#     df_final = pd.concat([
#         df_normal.sample(n=min_size, random_state=42),
#         df_drug.sample(n=min_size, random_state=42)
#     ]).sample(frac=1, random_state=42).reset_index(drop=True)

#     print(f"[Step 2] 데이터 규모: {len(df_final):,}건")

#     # 수사관님이 지시하신 '실전 수사용 BERT 베이스 모델' 유지
#     model_name = "klue/bert-base"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
#     model.to(DEVICE)
#     print("[System] 안정성 최우선 CPU 모드 가동 (GPU 에러 원천 차단)")

#     def tokenize_fn(examples):
#         return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

#     dataset = Dataset.from_pandas(df_final).train_test_split(test_size=0.1)
#     tokenized_ds = dataset.map(tokenize_fn, batched=True, desc="토큰화")

#     # CPU 환경에 맞춘 최적화 훈련 세팅
#     training_args = TrainingArguments(
#         output_dir="./DRUG_DETECTION_LOGS",
#         eval_strategy="steps",
#         eval_steps=200, 
#         save_strategy="steps",
#         save_steps=200,
#         learning_rate=2e-5,
#         per_device_train_batch_size=8, # 메모리 넉넉하므로 8로 세팅
#         per_device_eval_batch_size=16,
#         num_train_epochs=4, # 5번에서 4번으로 줄여서 학습 시간 단축 (성능 보장)
#         weight_decay=0.01,
#         load_best_model_at_end=True,
#         metric_for_best_model="f1",
#         logging_steps=50,
#         report_to="none",
#         use_cpu=True # 명시적으로 CPU 사용 선언 (UR error 방지)
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_ds["train"],
#         eval_dataset=tokenized_ds["test"],
#         compute_metrics=lambda p: {
#             'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1)),
#             'f1': f1_score(p.label_ids, p.predictions.argmax(-1))
#         },
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
#     )

#     print(f"[Step 3] 학습 시작 (장치: {DEVICE})")
#     trainer.train()

#     # 학습 완료 후 수사 모델 최종 저장
#     final_path = "./FINAL_INVESTIGATION_MODEL"
#     trainer.save_model(final_path)
#     tokenizer.save_pretrained(final_path)
#     print(f"[Step 4] 완료: {os.path.abspath(final_path)}")

# if __name__ == "__main__":
#     run_training()


import os
import glob
import json
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

DEVICE = torch.device("cpu")

def collect_data(base_path):
    print(f"[Step 1] 경로 스캔: {base_path}")
    target_pattern = os.path.join(base_path, "**/02.라벨링데이터/**/*.*")
    all_files = glob.glob(target_pattern, recursive=True)
    normal_texts = set()

    for f in tqdm(all_files, desc="정상 데이터 수집"):
        if os.path.isdir(f) or f.endswith('.zip'): continue
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext == '.json':
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    def find_text(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if k in ['utterance', 'text', 'sentence', 'form', 'content']:
                                    if isinstance(v, str) and len(v.strip()) > 1:
                                        normal_texts.add(v.strip())
                                else: find_text(v)
                        elif isinstance(obj, list):
                            for item in obj: find_text(item)
                    find_text(data)
            elif ext == '.csv':
                df = pd.read_csv(f)
                col = 'text' if 'text' in df.columns else df.columns[0]
                normal_texts.update(df[col].dropna().astype(str).tolist())
        except: continue
    return pd.DataFrame({'text': list(normal_texts), 'label': 0})

def run_training():
    source_path = r"C:\Users\82109\Downloads\141.한국어 멀티세션 대화\01-1.정식개방데이터"
    drug_path = "2026_DRUG_TRAIN_DATA_NATIONAL.csv"
    cache_path = "normal_data_cache.csv"

    # 1. 정상 데이터 로드
    if os.path.exists(cache_path):
        df_normal = pd.read_csv(cache_path)
    else:
        df_normal = collect_data(source_path)
        df_normal.to_csv(cache_path, index=False, encoding='utf-8-sig')

    # 2. 마약 데이터 로드 및 [치명적 버그 수정]
    df_drug = pd.read_csv(drug_path)
    
    # [핵심] 어떤 이름의 열이든 무조건 'text'로 이름을 강제 변경하여 데이터 누락(NaN) 방지
    drug_col = 'text' if 'text' in df_drug.columns else df_drug.columns[0]
    df_drug = df_drug.rename(columns={drug_col: 'text'})
    df_drug['label'] = 1

    # 결측치(빈칸) 완벽 제거
    df_drug = df_drug.dropna(subset=['text'])
    df_drug = df_drug[df_drug['text'].astype(str).str.strip() != '']
    df_normal = df_normal.dropna(subset=['text'])
    df_normal = df_normal[df_normal['text'].astype(str).str.strip() != '']

    # 3. 데이터 밸런싱 (빠르고 확실한 학습을 위해 총 1만 건으로 압축)
    min_size = min(len(df_normal), len(df_drug), 5000)
    df_final = pd.concat([
        df_normal.sample(n=min_size, random_state=42),
        df_drug.sample(n=min_size, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[Step 2] 진짜 데이터 학습 준비 완료: 총 {len(df_final):,}건")

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    dataset = Dataset.from_pandas(df_final).train_test_split(test_size=0.1)
    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="./DRUG_DETECTION_LOGS",
        eval_strategy="steps",
        eval_steps=100, 
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3, # 빠른 완료를 위해 3 Epoch으로 조정
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=lambda p: {
            'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1)),
            'f1': f1_score(p.label_ids, p.predictions.argmax(-1))
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f"[Step 3] 진짜 마약 문맥 학습 시작 ")
    trainer.train()

    final_path = "./FINAL_INVESTIGATION_MODEL"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"[Step 4] 수사 모델 재생성 완료: {os.path.abspath(final_path)}")

if __name__ == "__main__":
    run_training()