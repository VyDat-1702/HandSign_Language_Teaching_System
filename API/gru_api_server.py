from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from decimal import Decimal
from collections import deque
import mediapipe as mp  
import time
from mp import *
import os
os.environ["OMP_NUM_THREADS"] = "4"
# Khởi tạo FastAPI
app = FastAPI()
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Tải model YOLO chỉ một lần
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#khai báo các classes
class_names = np.array( ["aw","ee","oo","sac", "hoi","nang","nothing","aa","ow","uw", "nga","huyen"])

# log_dir = os.path.join('Logs_file\Logs_8_5\Logs_GRU_8_5')
# tb_callback = TensorBoard(log_dir=log_dir)
# cấu trúc model LSTM
'''
STEP 3: CREATE MODEL CLASS
'''
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(GRUModel, self).__init__()
        
        # GRU layer 1
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size1, 
                          batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # GRU layer 2
        self.gru2 = nn.GRU(input_size=hidden_size1, hidden_size=hidden_size2, 
                          batch_first=True)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Flatten (handled in forward)
        # Dense layer
        self.fc = nn.Linear(hidden_size2 * 30, num_classes)  # 30 là seq_len
        
        # Softmax không cần khai báo riêng vì thường xử lý trong loss function
        
    def forward(self, x):
        # GRU layer 1
        out, _ = self.gru1(x)  # out: [batch_size, seq_len, hidden_size1]
        out = self.dropout1(out)
        
        # GRU layer 2
        out, _ = self.gru2(out)  # out: [batch_size, seq_len, hidden_size2]
        out = self.dropout2(out)
        
        # Flatten
        out = out.reshape(out.size(0), -1)  # [batch_size, seq_len * hidden_size2]
        
        # Dense layer
        out = self.fc(out)  # [batch_size, num_classes]
        
        return out  # Softmax sẽ được áp dụng trong loss (CrossEntropyLoss)
# Khởi tạo mô hình
input_size = 63    # Số đặc trưng tại mỗi timestep
hidden_size1 = 256 # Kích thước hidden state của GRU đầu tiên
hidden_size2 = 128 # Kích thước hidden state của GRU thứ hai
num_classes = class_names.shape[0]  # Số lớp đầu ra (từ biến actions_train)

model = GRUModel(input_size, hidden_size1, hidden_size2, num_classes)  
#compile model 
# max_epoch = 120
# LR = 0.001
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
torch.manual_seed(1)
'''
STEP 4: INSTANTIATE MODEL CLASS
'''




#######################
#  USE GPU FOR MODEL  #
#######################
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load trọng số mô hình từ file đã lưu
model.load_state_dict(torch.load('D:/install/AI/Code/Project/GRU_MODEL/gru_model.pth'))

# Đặt mô hình vào chế độ đánh giá (evaluation mode)
model.eval()

# Thông số đầu vào
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.5  # Ngưỡng dự đoán 50%


templates = Jinja2Templates(directory="templates")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Hàm đọc camera async
# async def read_camera():
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(None, cap.read)

@app.websocket("/ws")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
            while True:
                contents = await websocket.receive_bytes()  # Nhận bytes từ WebSocket
                keypoints = np.frombuffer(contents, dtype=np.float32)  # Decode thành mảng float32
                print("Keypoints shape before reshape:", keypoints.shape)

                # Chuyển thành list 2D có shape (30, 63)
                keypoints = keypoints.reshape(30, 63).tolist()
                print("Keypoints shape after reshape:", len(keypoints), len(keypoints[0]))  # Phải ra (30, 63)


                # 🔴 Dự đoán
                sequence_tensor = np.expand_dims(keypoints, axis=0)  # Thêm chiều batch
                sequence_tensor = torch.tensor(sequence_tensor, dtype=torch.float32).to(device)
                with torch.no_grad():
        # Dự đoán
                            res_tensor = model(sequence_tensor)  # Gọi mô hình trực tiếp
                            res = torch.softmax(res_tensor[0], dim=-1).cpu().numpy()  # Chuyển logits thành xác suất
                            predicted_index = np.argmax(res)
                            prediction_score = res[predicted_index]
                            print(prediction_score)

                
                # Encode ảnh và gửi qua WebSocket
                data = np.array([predicted_index, prediction_score], dtype=np.float32)
                await websocket.send_bytes(data.tobytes())  # 8 bytes nếu 2 phần tử

                # Giảm tải CPU/GPU
                await asyncio.sleep(0.01)  # Có thể chỉnh 0.01 - 0.05 tùy tốc độ cần thiết

    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
