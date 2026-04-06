import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QFileDialog
from src.preprocessing import DataPreprocessor
from src.nlp_model import KoBERTModel
from src.graph_model import GNNModel
from src.visualization import GraphVisualizer

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("이상 행위 탐지 시스템 (Dongseo Univ.)")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: black; color: #00FF00; font-family: Consolas;")
        self.log_console.append("[System] 시스템 대기 중")
        layout.addWidget(self.log_console)
        
        # 파일 업로드
        self.btn_upload = QPushButton("1. 분석할 로그 파일 업로드 (.txt, .csv, .exe)", self)
        self.btn_upload.clicked.connect(self.upload_file)
        layout.addWidget(self.btn_upload)
        
        # 데이터 전처리
        self.btn_preprocess = QPushButton("2. 데이터 전처리 및 텐서 변환 (Pandas/NumPy)", self)
        self.btn_preprocess.clicked.connect(self.run_preprocessing)
        layout.addWidget(self.btn_preprocess)
        
        # AI 분석 실행
        self.btn_ai = QPushButton("3. AI 분석 실행 (KoBERT & GNN)", self)
        self.btn_ai.clicked.connect(self.run_ai_models)
        layout.addWidget(self.btn_ai)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # 상태 저장용 변수
        self.selected_file_path = None
        self.preprocessor = None

    def upload_file(self):
        """윈도우 파일 탐색기를 열어 파일을 선택하게 하는 함수입니다."""
        self.log_console.append("\n====================================")
        # QFileDialog를 이용해 모든 파일또는 특정 확장자를 선택할 수 있는 팝업 띄우기
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "분석할 혐의자 로그 파일 선택", 
            "", 
            "All Files (*);;Text Files (*.txt);;CSV Files (*.csv);;Executable Files (*.exe)"
        )
        
        if file_path:
            self.selected_file_path = file_path
            self.log_console.append(f"[Success] 타겟 파일이 업로드되었습니다:\n -> {file_path}")
        else:
            self.log_console.append("[System] 파일 선택이 취소되었습니다.")

    def run_preprocessing(self):
        self.log_console.append("\n====================================")
        # 수사관이 파일을 안 올리면 팅
        if not self.selected_file_path:
            self.log_console.append("[Error] 먼저 1번 버튼을 눌러 분석할 파일을 업로드")
            return
            
        self.log_console.append("[Process] 2. 데이터 전처리를 시작합니다...")
        # 전처리기에 업로드한 파일 경로 넣기
        self.preprocessor = DataPreprocessor(raw_file_path=self.selected_file_path)
        self.preprocessor.load_data()
        self.log_console.append("[Success] Pandas 파싱 및 행위 Tensor 변환 완료!")

    def run_ai_models(self):
        self.log_console.append("\n====================================")
        if self.preprocessor is None or self.preprocessor.df is None:
            self.log_console.append("[Error] 먼저 2번 데이터 전처리를 실행해주세요!")
            return
            
        self.log_console.append("[Process] 3. AI 트랙 가동 중...")
        
        behavior_score = self.preprocessor.to_tensor()
        
        nlp = KoBERTModel(self.preprocessor.df)
        nlp_score = nlp.get_context_score()
        self.log_console.append("  -> KoBERT 문맥 분석 완료")
        
        gnn = GNNModel(self.preprocessor.df, nlp_score, behavior_score)
        result_dict = gnn.run_message_passing()
        self.log_console.append(f"  -> PyTorch GNN 최종 분류 완료: {result_dict}")
        
        self.log_console.append("[Process] NetworkX 조직도 렌더링 중...")
        viz = GraphVisualizer(result_dict)
        viz.draw_network()
        self.log_console.append("[Success] 모든 분석이 완료되었습니다! (팝업 창을 확인하세요)")

def run_gui():
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())