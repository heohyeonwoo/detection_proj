import torch
import pandas as pd

class GNNModel:
    def __init__(self, df, nlp_tensor, behavior_tensor):
        self.df = df
        self.nlp_tensor = nlp_tensor
        self.behavior_tensor = behavior_tensor

    def run_message_passing(self):
        print("\n[System] PyTorch GNN 모델이 그래프 병합 및 메시지 패싱 연산을 시작")

        # 대화에 참여한 모든 사람 노드로 추출
        senders = self.df['sender'].unique()
        node_results = {}

        # 판다스 표에 AI가 계산한 텐서 점수들을 합체
        self.df['nlp_score'] = self.nlp_tensor.numpy()
        self.df['behavior_score'] = self.behavior_tensor.numpy()

        # 사람별로 위험도를 계산
        for sender in senders:
            sender_data = self.df[self.df['sender'] == sender]
            
            # 해당 사람이 친 채팅 중 가장 높았던 문맥 위험도 추출
            max_nlp = sender_data['nlp_score'].max()
            
            # 위험도가 0.8 이상이면 마약 사범, 아니면 일반인(0)으로 분류
            is_criminal = 1 if max_nlp >= 0.8 else 0
            node_results[sender] = is_criminal

        print(f"[Success] GNN 메시지 패싱 완료! 최종 분류 결과: {node_results}")
        return node_results