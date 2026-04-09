import os
import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QFileDialog
from PyQt6 import QtGui 
from google import genai

# DLL 에러 없앨려면 하래 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if os.name == 'nt':
    torch_lib_path = r'C:\Users\82109\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
        os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")

from src.preprocessing import DataPreprocessor
from src.nlp_model import KoBERTModel
from src.graph_model import GNNModel
from src.visualization import GraphVisualizer

# 제미나이 API 요약 기능 넣어 볼려고 써보는중 ..
GEMINI_API_KEY = "AIzaSyDkgoXZKhIKkQOvd73Uzd6_jCCxkI9GtmI" 

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("마약 범죄 실시간 이상 행위 탐지 시스템")
        self.setGeometry(100, 100, 1000, 800)
        
        layout = QVBoxLayout()
        
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: black; color: #00FF00; font-family: Consolas; font-size: 13px;")
        self.log_console.append("[System] 시스템 대기 중...")
        layout.addWidget(self.log_console)
        
        self.btn_upload = QPushButton("1. 분석할 대상 로그 파일 업로드 (.txt, .csv)", self)
        self.btn_upload.setMinimumHeight(45)
        self.btn_upload.clicked.connect(self.upload_file)
        layout.addWidget(self.btn_upload)
        
        self.btn_preprocess = QPushButton("2. 전체 데이터 전처리 및 대화 흐름 진짜 요약", self)
        self.btn_preprocess.setMinimumHeight(45)
        self.btn_preprocess.clicked.connect(self.run_preprocessing)
        layout.addWidget(self.btn_preprocess)
        
        self.btn_ai = QPushButton("3. AI 분석 가동 및 위험 대화 증거 추출", self)
        self.btn_ai.setMinimumHeight(45)
        self.btn_ai.clicked.connect(self.run_ai_models)
        layout.addWidget(self.btn_ai)
        
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.selected_file_path = None
        self.preprocessor = None

    def scroll_to_bottom(self):
        self.log_console.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def upload_file(self):
        self.log_console.append("\n" + "="*60)
        file_path, _ = QFileDialog.getOpenFileName(self, "분석할 혐의자 로그 파일 선택", "", "All Files (*);;Text Files (*.txt);;CSV Files (*.csv)")
        if file_path:
            self.selected_file_path = file_path
            self.log_console.append(f"[Success] 타겟 파일이 업로드되었습니다:\n -> {file_path}")
        else:
            self.log_console.append("[System] 파일 선택이 취소되었습니다.")
        self.scroll_to_bottom()

    def run_preprocessing(self):
        self.log_console.append("\n" + "="*60)
        if not self.selected_file_path:
            self.log_console.append("[Error] 먼저 1번 버튼을 눌러 분석할 파일을 업로드하세요.")
            self.scroll_to_bottom()
            return
            
        self.log_console.append("[Process] 2. 전체 데이터 전처리를 시작합니다...")
        self.preprocessor = DataPreprocessor(target_path=self.selected_file_path)
        self.preprocessor.load_data()
        
        df = self.preprocessor.df
        if df is not None and not df.empty:
            total_rows = len(df)
            self.log_console.append(f"[Success] 총 {total_rows}건의 대화가 누락 없이 100% 로드되었습니다.")
            
            self.log_console.append("\n--- 전체 대화 내역 (모두 출력) ---")
            pd.set_option('display.max_rows', None)
            all_data_str = df[['datetime', 'sender', 'message']].to_string(index=False)
            self.log_console.append(all_data_str)
            self.log_console.append("----------------------------------")

            self.log_console.append("\n[Process] 전체 대화 흐름을 분석하여 요약을 작성 중입니다...")
            QApplication.processEvents() 
            
            summary_text = ""
            try:
                if GEMINI_API_KEY != "여기에_발급받은_키를_붙여넣으세요":
                    # 최신 모델 
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    
                    filtered_df = df[~df['sender'].str.contains("방장봇|봇|시스템|참여", na=False)]
                    chat_log = "\n".join(filtered_df['message'].astype(str).tolist()[-1200:]) 
                    
                    prompt = (
                        "너는 범죄 수사관을 돕는 데이터 요약 전문가야. 아래의 대화 로그를 읽고, "
                        "이 방에서 사람들이 전체적으로 어떤 흐름으로 대화했는지, "
                        "그리고 수사관이 참고해야 할 특이한 점이 있는지 3~4줄의 문장으로 정성껏 요약해줘.\n\n" + chat_log
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    summary_text = f" -  [AI 대화 흐름 심층 분석]:\n   {response.text.strip().replace('\n', '\n   ')}"
                else:
                    summary_text = " -  요약 실패: API 키가 입력되지 않았습니다."
            except Exception as e:
                summary_text = f" -  요약 엔진 연결 실패: 구글 API 상태를 확인하세요. ({e})"

            start_date = df['datetime'].min().strftime('%Y-%m-%d')
            end_date = df['datetime'].max().strftime('%Y-%m-%d')
            users_count = df['sender'].nunique()
            top_talker = df['sender'].value_counts().index[0]
            
            summary = (
                f"\n===================================\n"
                f"   [수사 데이터 기초 요약 리포트]\n"
                f" - 수집 기간: {start_date} ~ {end_date}\n"
                f" - 총 대화 건수: {len(df)}건\n"
                f" - 톡방 참여 인원: {users_count}명\n"
                f" - 최다 발언자(주동자 의심): {top_talker}\n"
                f"{summary_text}\n"
                f"===================================\n"
            )
            self.log_console.append(summary)
            self.scroll_to_bottom() 
        else:
            self.log_console.append("[Error] 데이터 로드에 실패했습니다.")
            self.scroll_to_bottom()

    def run_ai_models(self):
        self.log_console.append("\n" + "="*60)
        if self.preprocessor is None or self.preprocessor.df is None:
            self.log_console.append("[Error] 먼저 2번 데이터 전처리를 실행해주세요")
            self.scroll_to_bottom()
            return
            
        self.log_console.append("[Process] 3. AI 분석 가동 중 ")
        self.scroll_to_bottom()
        QApplication.processEvents()
        
        behavior_score = self.preprocessor.to_tensor()
        nlp = KoBERTModel(self.preprocessor.df)
        
        messages_list = self.preprocessor.df['message'].tolist()
        nlp_score = nlp.get_context_score(messages=messages_list)
        
        self.preprocessor.df['nlp_score'] = nlp_score.numpy()
        
        gnn = GNNModel(self.preprocessor.df, nlp_score, behavior_score)
        result_dict = gnn.run_message_passing()
        
        if not self.preprocessor.df.empty:
            top_suspect = self.preprocessor.df.groupby('sender')['nlp_score'].max().idxmax()
            result_dict[top_suspect] = 1  
            
        suspects_list = [person for person, is_criminal in result_dict.items() if is_criminal == 1]
        self.log_console.append(f"\n 최종 탐지된 혐의자: {suspects_list}")

        self.log_console.append("\n[결정적 증거] 시스템이 탐지한 모든 위험 대화 내역")
        
        evidence_df = self.preprocessor.df[self.preprocessor.df['nlp_score'] > 0.1].sort_values(by='nlp_score', ascending=True)
        
        if not evidence_df.empty:
            for _, row in evidence_df.iterrows():
                self.log_console.append(f" * ({row['datetime']}) [{row['sender']}] : {row['message']} (위험도: {row['nlp_score']:.2f})")
        else:
            self.log_console.append(" 위험 징후가 발견된 대화가 없습니다.")
        
        self.log_console.append("\n[Process] 마약 조직망 엣지/노드 시각화 렌더링 중...")
        self.scroll_to_bottom()
        
        viz = GraphVisualizer(result_dict)
        viz.draw_network()
        
        self.log_console.append("[Success] 모든 분석이 완료되었습니다!")
        self.scroll_to_bottom()

def run_gui():
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())