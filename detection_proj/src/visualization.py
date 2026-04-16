import os
import webbrowser
import random
from pyvis.network import Network

class GraphVisualizer:
    def __init__(self, result_dict):
        self.result_dict = result_dict

    def draw_network(self):
        net = Network(height='100vh', width='100%', bgcolor='#0f0f13', font_color='white', select_menu=True)
        
        suspects = []
        normals = []

        for person, is_criminal in self.result_dict.items():
            safe_name = str(person)
            
            if is_criminal == 1:
                net.add_node(
                    safe_name, 
                    label=safe_name, 
                    title="혐의자 (마약 의심)", 
                    color={'background': '#ff2a2a', 'border': '#ffffff'},
                    borderWidth=3,
                    size=45,
                    font={'size': 22, 'color': 'white', 'strokeWidth': 5, 'strokeColor': 'black', 'face': 'sans-serif'}
                )
                suspects.append(safe_name)
            else:
                net.add_node(
                    safe_name, 
                    label=safe_name, 
                    title="일반 사용자", 
                    color={'background': '#2a75ff', 'border': '#88bbff'},
                    borderWidth=1,
                    size=15,
                    font={'size': 14, 'color': '#eeeeee', 'strokeWidth': 3, 'strokeColor': 'black'}
                )
                normals.append(safe_name)

        # GNN 뇌 구조 시냅스처럼 곡선 처리
        # 혐의자들끼리 진한 빨간색 
        for i in range(len(suspects)):
            for j in range(i+1, len(suspects)):
                net.add_edge(suspects[i], suspects[j], color='#ff4d4d', value=6, smooth={'type': 'curvedCW'})

        if suspects:
            for normal in normals:
                # 일반인 -> 혐의자 연결 
                targets = random.sample(suspects, k=min(2, len(suspects)))
                for target in targets:
                    net.add_edge(normal, target, color='rgba(150, 150, 150, 0.4)', value=1, smooth={'type': 'continuous'})
                
                # 일반인 -> 일반인 일상적 연결
                if random.random() > 0.5 and len(normals) > 2:
                    other_normal = random.choice([n for n in normals if n != normal])
                    net.add_edge(normal, other_normal, color='rgba(42, 117, 255, 0.15)', value=0.5, smooth={'type': 'continuous'})

        #  GNN 스타일 유기적 물리 엔진
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.015,
              "springLength": 120,
              "springConstant": 0.04
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 100
            }
          }
        }
        """)

        # 브라우저로 띄우기
        output_path = 'criminal_network_result.html'
        net.write_html(output_path)
        
        abs_path = os.path.abspath(output_path)
        try:
            safe_url = f"file:///{abs_path.replace(os.sep, '/')}"
            webbrowser.open(safe_url)
        except Exception as e:
            os.system(f'start "" "{abs_path}"')