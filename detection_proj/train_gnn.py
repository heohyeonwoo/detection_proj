import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

print("1. PyTorch & GPU 세팅 완료!")

# 진짜 뇌 구조 해야함......sibal
class DrugDetectionGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(DrugDetectionGNN, self).__init__()
        # 첫 번째 뇌세포 층 (노드의 특징을 16개 차원으로 확장)
        self.conv1 = GCNConv(num_features, 16)
        # 두 번째 뇌세포 층 (16개 차원을 최종 2개(0:정상, 1:범죄)로 압축 분류)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # 순전파(Forward) 연산 시작
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x) # 활성화 함수 (복잡한 패턴을 학습하게 해줌)
        F.dropout(x, training=self.training) # 과적합(오탐) 방지용 드롭아웃
        x = self.conv2(x, edge_index, edge_weight)
        
        return F.log_softmax(x, dim=1) # 확률값으로 변환하여 출력

# ==========================================
# [2] 학습용 가상 데이터셋 준비 (원래는 Pandas에서 가져옴)
# ==========================================
# 노드(사람) 4명의 특징: [KoBERT문맥점수, 프로필사진유무, 가입기간]
x = torch.tensor([[0.9, 1, 5],   # A: 위험문맥 씀
                  [0.1, 1, 10],  # B: 정상
                  [0.85, 0, 2],  # C: 위험문맥 씀
                  [0.05, 1, 30]], dtype=torch.float)

# 엣지(연결): 누가 누구랑 대화했는가? (0->1, 1->2 등)
edge_index = torch.tensor([[0, 1, 1, 2, 0, 2],
                           [1, 0, 2, 1, 2, 0]], dtype=torch.long)

# 엣지 가중치: Pandas/NumPy가 계산했던 '행위 위험도 (Behavior Score)'
edge_weight = torch.tensor([0.9, 0.9, 0.1, 0.1, 0.8, 0.8], dtype=torch.float)

# 정답지(Label): 수사관이 미리 매겨둔 진짜 정답 (0, 2번 사람은 마약사범=1 / 1, 3번은 정상=0)
y = torch.tensor([1, 0, 1, 0], dtype=torch.long)

# PyTorch Geometric 전용 데이터 팩으로 묶기
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

# ==========================================
# [3] 본격적인 학습 (Training) 셋업
# ==========================================
model = DrugDetectionGNN(num_features=3) # 특징이 3개이므로 3 입력
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 최적화 함수 (Adam)
criterion = torch.nn.NLLLoss() # 오차 계산 함수 (Loss)

print("\n2. GNN 모델 학습을 시작합니다! (총 200 Epoch)")
model.train()

# 200번의 반복 학습(Epoch) 시작!
for epoch in range(200):
    optimizer.zero_grad() # 1. 이전 미분값 초기화
    
    out = model(data)     # 2. 순전파 (예측해보기)
    loss = criterion(out, data.y) # 3. 오차 계산 (정답이랑 얼마나 틀렸나?)
    
    loss.backward()       # 4. 역전파 (거꾸로 돌아가며 오답 노트 작성)
    optimizer.step()      # 5. 가중치 업데이트 (뇌 구조 수정)
    
    # 20번마다 학습 진행 상황 출력
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:03d} | 오차(Loss): {loss.item():.4f}")

# ==========================================
# [4] 학습된 뇌(가중치)를 파일로 추출
# ==========================================
torch.save(model.state_dict(), 'gnn_weights.pt')
print("\n3. [Success] 학습 완료! 'gnn_weights.pt' 파일이 생성되었습니다.")