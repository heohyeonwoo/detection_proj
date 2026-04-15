import os
import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QFileDialog
from PyQt6 import QtGui 
from google import genai

# DLL 에러 방지
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

# 🔒 환경변수로 API 키 관리 (중요)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("마약 범죄 실시간 이상 행위 탐지 시스템")
        self.setGeometry(100, 100, 1000, 800)
        
        layout = QVBoxLayout()
        
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: black; color: #00FF00; font-family: Consolas; font-size: 13px;")
        self.log_console.append("[System] 시스템 초기화 완료. 대기 중...")
        layout.addWidget(self.log_console)
        
        self.btn_upload = QPushButton("1. 로그 파일 업로드 (.txt, .csv)", self)
        self.btn_upload.setMinimumHeight(45)
        self.btn_upload.clicked.connect(self.upload_file)
        layout.addWidget(self.btn_upload)
        
        self.btn_preprocess = QPushButton("2. 데이터 전처리 및 대화 흐름 분석", self)
        self.btn_preprocess.setMinimumHeight(45)
        self.btn_preprocess.clicked.connect(self.run_preprocessing)
        layout.addWidget(self.btn_preprocess)
        
        self.btn_ai = QPushButton("3. AI 분석 및 위험 탐지", self)
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
        file_path, _ = QFileDialog.getOpenFileName(self, "로그 파일 선택", "", "All Files (*);;Text Files (*.txt);;CSV Files (*.csv)")
        
        if file_path:
            self.selected_file_path = file_path
            self.log_console.append(f"[Success] 파일 업로드 완료:\n → {file_path}")
        else:
            self.log_console.append("[System] 파일 선택 취소")
        
        self.scroll_to_bottom()

    def run_preprocessing(self):
        self.log_console.append("\n" + "="*60)

        if not self.selected_file_path:
            self.log_console.append("[Error] 파일을 먼저 업로드하세요.")
            return

        self.log_console.append("[Process] 데이터 전처리 시작...")
        self.preprocessor = DataPreprocessor(target_path=self.selected_file_path)
        self.preprocessor.load_data()

        df = self.preprocessor.df

        if df is None or df.empty:
            self.log_console.append("[Error] 데이터 로드 실패")
            return

        self.log_console.append(f"[Success] 총 {len(df)}건 데이터 로드 완료")

        # 🔍 전체 출력
        self.log_console.append("\n[전체 대화 로그]")
        self.log_console.append(df[['datetime', 'sender', 'message']].to_string(index=False))

        # 🤖 Gemini 요약
        summary_text = ""
        try:
            if GEMINI_API_KEY:
                client = genai.Client(api_key=GEMINI_API_KEY)

                filtered_df = df[~df['sender'].str.contains("봇|시스템", na=False)]
                chat_log = "\n".join(filtered_df['message'].astype(str).tolist()[-500:])

                prompt = (
                    "다음 대화를 분석하여 수사 관점에서 핵심 흐름과 이상 징후를 3줄로 요약하라:\n\n"
                    + chat_log
                )

                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )

                summary_text = response.text.strip()
            else:
                summary_text = "API 키 없음 → 요약 생략"

        except Exception as e:
            summary_text = f"요약 실패: {e}"

        # 📊 요약 리포트
        summary = f"""
==============================
[수사 요약 리포트]
- 기간: {df['datetime'].min()} ~ {df['datetime'].max()}
- 총 대화: {len(df)}건
- 참여자: {df['sender'].nunique()}명
- 주요 사용자: {df['sender'].value_counts().idxmax()}

[AI 요약]
{summary_text}
==============================
"""
        self.log_console.append(summary)
        self.scroll_to_bottom()

    def run_ai_models(self):
        self.log_console.append("\n" + "="*60)

        if self.preprocessor is None:
            self.log_console.append("[Error] 전처리 먼저 실행")
            return

        self.log_console.append("[Process] AI 분석 시작...")
        QApplication.processEvents()

        df = self.preprocessor.df

        behavior_score = self.preprocessor.to_tensor()
        nlp = KoBERTModel(df)

        messages_list = df['message'].tolist()
        context_tensor, pattern_tensor = nlp.get_context_score(messages_list)

        # 📊 점수 계산
        df['nlp_score'] = context_tensor.numpy()
        df['pattern_score'] = pattern_tensor.numpy()
        df['final_score'] = df['nlp_score'] * 0.7 + df['pattern_score'] * 0.3

        # 🔗 GNN 분석
        gnn = GNNModel(df, context_tensor, behavior_score)
        result_dict = gnn.run_message_passing()

        # 🧠 주요 용의자
        top_suspect = df.groupby('sender')['final_score'].max().idxmax()
        result_dict[top_suspect] = 1

        suspects = [k for k, v in result_dict.items() if v == 1]

        self.log_console.append(f"\n[결과] 탐지된 혐의자: {suspects}")

        # 🔥 증거 출력
        self.log_console.append("\n[증거 로그]")
        evidence_df = df[df['final_score'] > 0.3].sort_values(by='final_score', ascending=False)

        if evidence_df.empty:
            self.log_console.append("위험 대화 없음")
        else:
            for _, row in evidence_df.iterrows():
                self.log_console.append(
                    f"[{row['datetime']}] {row['sender']} → {row['message']} (위험도: {row['final_score']:.2f})"
                )

        # 📊 그래프 시각화
        self.log_console.append("\n[Process] 네트워크 시각화 실행...")
        GraphVisualizer(result_dict).draw_network()

        self.log_console.append("[Success] 분석 완료")
        self.scroll_to_bottom()


def run_gui():
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())