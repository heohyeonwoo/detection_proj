import os
import sys
import ctypes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

torch_lib_path = r'C:\Users\82109\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib'

if os.name == 'nt' and os.path.exists(torch_lib_path):
    # 폴더 탐색 순위 1순위로 강제 지정
    os.add_dll_directory(torch_lib_path)
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
    
    try:
        ctypes.CDLL(os.path.join(torch_lib_path, 'c10.dll'))
        ctypes.CDLL(os.path.join(torch_lib_path, 'torch_cpu.dll'))
    except Exception:
        pass

import numpy as np

from gui.main_window import run_gui

if __name__ == "__main__":
    run_gui()