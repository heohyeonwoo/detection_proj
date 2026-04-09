import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from src.preprocessing import DataPreprocessor
from src.nlp_model import KoBERTModel
import pandas as pd
import os

print("[1] 전체 원본 데이터 전처리 및 분석 시작")
preprocessor = DataPreprocessor()
preprocessor.load_data()
behavior_tensor = preprocessor.to_tensor()

nlp = KoBERTModel(preprocessor.df)
nlp_tensor = nlp.get_context_score()

df = preprocessor.df
df['nlp_score'] = nlp_tensor.numpy()
df['behavior_score'] = behavior_tensor.numpy()

# 문자열 시간을 Pandas DateTime 객체로 변환
df['datetime'] = pd.to_datetime(df['datetime'])


print("[2] 전체 데이터 동적 그래프 스냅샷 생성 중...")
# 전체 데이터를 10분 단위 시간 윈도우로 분할하여 동적 그래프 구성
snapshots = []
df.set_index('datetime', inplace=True)
grouped = df.groupby(pd.Grouper(freq='10min'))

for time_window, group in grouped:
    if group.empty:
        continue
    
    senders = group['sender'].unique()
    node_features = []
    labels = []

    for sender in senders:
        sender_data = group[group['sender'] == sender]
        # 최대 문맥 위험도와 평균 행위 위험도
        max_nlp = sender_data['nlp_score'].max()
        avg_behavior = sender_data['behavior_score'].mean()
        node_features.append([max_nlp, avg_behavior])
        
        # 위험도가 0.8 이상인 노드를 범죄자(1)로 라벨링
        labels.append(1 if max_nlp >= 0.8 else 0)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # 엣지 연결
    edges = []
    for idx1 in range(len(senders)):
        for idx2 in range(idx1 + 1, len(senders)):
            edges.append([idx1, idx2])
            edges.append([idx2, idx1])
    
    # 윈도우 내에 혼자만 말한 경우
    if not edges: 
        for idx in range(len(senders)):
            edges.append([idx, idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor([1.0] * len(edges), dtype=torch.float)

    snapshot_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    snapshots.append(snapshot_data)

print(f"[Success] 전체 데이터셋 기반, 총 {len(snapshots)}개의 10분 단위 동적 그래프 스냅샷이 생성되었습니다!")
print("[3] GNN 실전 학습 (얼리스토핑 적용)")


class DrugDetectionGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(DrugDetectionGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return F.log_softmax(self.conv2(x, edge_index, edge_weight), dim=1)

model = DrugDetectionGNN(num_features=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# 딥러닝 하이퍼파라미터 설정
MAX_EPOCHS = 2000
PATIENCE = 50  # 50번 동안 오차가 개선되지 않으면 조기 종료
best_loss = float('inf')
patience_counter = 0

model.train()
for epoch in range(MAX_EPOCHS):
    total_loss = 0
    
    # 생성된 모든 10분 단위 스냅샷을 순회하며 전체 데이터 학습
    for data in snapshots:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(snapshots)

    # 얼리스토핑(Early Stopping) 및 최적 가중치 저장 로직
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0  # 신기록 달성 시 인내심 카운터 초기화
        os.makedirs('models/gnn_weights', exist_ok=True)
        torch.save(model.state_dict(), 'models/gnn_weights/real_gnn_best.pt')
    else:
        patience_counter += 1

    # 진행 상황 모니터링
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:04d} | 평균 오차(Loss): {avg_loss:.6f} | 인내심 카운터: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"\n [Early Stopping] {epoch+1}번째 에포크에서 학습이 완벽히 안정화되어 훈련을 강제 종료합니다!")
        break

print("\n[Success] 학습 완료! 과적합을 방지한 최적의 뇌가 'models/gnn_weights/real_gnn_best.pt'에 안전하게 저장되었습니다!")