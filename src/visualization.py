import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class GraphVisualizer:
    def __init__(self, node_results):
        self.node_results = node_results
        self.graph = nx.Graph()

    def draw_network(self):
        print("\n[System] NetworkX가 최종 조직도를 렌더링합니다...")

        if not self.node_results:
            print("[Error] 그릴 데이터가 없습니다.")
            return

        # 노드 색상 지정 (범죄 1=빨강, 정상 0=파랑)
        color_map = []
        for node, is_criminal in self.node_results.items():
            self.graph.add_node(node)
            if is_criminal == 1:
                color_map.append('red')
            else:
                color_map.append('skyblue')

        # 같은 채팅방에 있으므로 엣지로 연결
        nodes = list(self.node_results.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.graph.add_edge(nodes[i], nodes[j])

        # 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 팝업 창
        plt.figure(figsize=(8, 6))
        plt.title("마약 범죄 조직망 탐지 위상 그래프 (Red: 위험, Blue: 정상)", fontsize=15, fontweight='bold')
        
        pos = nx.spring_layout(self.graph) # 거미줄 레이아웃
        nx.draw(self.graph, pos, node_color=color_map, with_labels=True, node_size=3000, font_size=12, font_weight='bold', edge_color='gray', font_family='Malgun Gothic')

        print("[Success] 모니터에 시각화 창을 성공적으로 띄웠습니다!")
        plt.show()