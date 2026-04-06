import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from src.preprocessing import DataPreprocessor
from src.nlp_model import KoBERTModel
import os

print("===========================================")
print("[1] 카카오톡 원본 데이터 전처리 및 분석 시작")
print("===========================================")
# 1. 우리가 만든 파이프라인으로 진짜 데이터 불러오기
preprocessor = DataPreprocessor()
preprocessor.load_data()
behavior_tensor = preprocessor.to_tensor()

nlp = KoBERTModel(preprocessor.df)
nlp_tensor = nlp.get_context_score()

# 2. 표에 점수 합체
df = preprocessor.df
df['nlp_score'] = nlp_tensor.numpy()
df['behavior_score'] = behavior_tensor.numpy()

print("\n===========================================")
print("[2] GNN용 위상 그래프(Graph) 데이터 생성 중...")
print("===========================================")
# 3. GNN이 학습할 수 있는 노드(사람)와 엣지(관계)로 변환
senders = df['sender'].unique()
node_features = []
labels = []

for sender in senders:
    sender_data = df[df['sender'] == sender]
    # 각 사람의 '최고 문맥 위험도'와 '평균 행위 위험도'를 특징(Feature)으로 사용
    max_nlp = sender_data['nlp_score'].max()
    avg_behavior = sender_data['behavior_score'].mean()
    node_features.append([max_nlp, avg_behavior])
    
    # 정답지(Label) 자동 생성: 문맥 점수가 높은 사람(허현우)을 범죄자(1)로 라벨링
    labels.append(1 if max_nlp >= 0.8 else 0)

x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

# 엣지 연결 (두 사람이 같은 채팅방에 있으므로 서로 연결)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) 
edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

print("\n===========================================")
print("[3] 진짜 카카오톡 데이터 GNN 학습(Training) 시작!")
print("===========================================")

# 4. GNN 모델 아키텍처
class DrugDetectionGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(DrugDetectionGNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return F.log_softmax(self.conv2(x, edge_index, edge_weight), dim=1)

model = DrugDetectionGNN(num_features=2) # 특징이 2개(문맥, 행위)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = torch.nn.NLLLoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:03d} | 오차(Loss): {loss.item():.4f}")

# 5. 완성된 진짜 뇌(가중치)를 지정된 폴더에 저장
os.makedirs('models/gnn_weights', exist_ok=True)
torch.save(model.state_dict(), 'models/gnn_weights/real_gnn.pt')
print("\n[Success] 학습 완료! 진짜 뇌가 'models/gnn_weights/real_gnn.pt'에 저장되었습니다!")