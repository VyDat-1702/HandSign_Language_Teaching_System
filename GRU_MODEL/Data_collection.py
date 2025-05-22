import cv2
import numpy as np 
import os
import mediapipe as mp
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mp_holistic = mp.solutions.holistic # modul nhận dạng face mesh, hand landmark, pose
mp_drawing = mp.solutions.drawing_utils #dùng để vẽ và connection

#Tiền xử lý ảnh đầu vào và chạy mô hình MediaPipe để phát hiện keypoints
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#mô hình xử lý ảnh RGB
    image.flags.writeable = False #không thực hiện thao tác ghi
    result = model.process(image)#xử lý để phát hiện keypoints
    image.flags.writeable = True #bật ghi
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)#chuyển ảnh lại định dạng ban đầu là BGR
    return image, result

#vẽ các keypoints
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, #vẽ mốc và đường nối
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
    
    
 #Trích xuất toạ độ các keypoints của bàn tay phải thành một vector 1 chiều   
def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return rh #[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

if __name__ == "__main__":
    DATA_TRAIN_PATH = os.path.join("MP_Data_Train")
    VIDEO_TRAIN_PATH = os.path.join("MP_Video_Train")

    actions = np.array(["aw", "ee", "ow", "sac", "hoi", "nang", "nothing", "aa", "oo", "uw", "nga", "huyen"])

    no_sequences_train = 170
    no_sequences_test = 20
    sequence_length = 30
    start_folder = 0

    # Tạo thư mục lưu
    for action in actions:
        for sequence in range(no_sequences_train):
            os.makedirs(os.path.join(DATA_TRAIN_PATH, action, str(sequence)), exist_ok=True)
            os.makedirs(os.path.join(VIDEO_TRAIN_PATH, action, str(sequence)), exist_ok=True)

    # Khởi tạo camera và các cài đặt
    cap = cv2.VideoCapture(0)
    w, h = int(cap.get(3)), int(cap.get(4)) #id3: wighth, id4: heigh
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #Four character code là một mã định dạng video xác định decode để lưu video

    if not cap.isOpened():
        print("Cannot open camera.")
        exit()

    # Thu thập dữ liệu
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #min detection confidence là độ tự tin tối thiểu để mô hình có thể nhận dạng, 
        #min tracking confidence là độ tự tin tối thiểu mà mô hình cần có để theo dõi một đối tượng sau khi đã phát hiện
        for action in actions:
            for sequence in range(start_folder, start_folder + no_sequences_train):
                wv_path = os.path.join(VIDEO_TRAIN_PATH, action, str(sequence), f"{sequence}.mp4")
                VDWT = cv2.VideoWriter(wv_path, fourcc, 30, (w, h))

                if not VDWT.isOpened():
                    print(f"Error opening VideoWriter: {wv_path}")
                    continue

                print(f"Press 'q' to start recording: {wv_path}")
                while True:
                    ret, frame = cap.read() #đọc cap để hiện thị trực tiếp, cho phép xem những gì đang thu thập thực tế
                    if not ret:
                        print("Error reading camera.")
                        break
                    cv2.putText(frame, f"Recording action: {action}, video {sequence}", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.imshow('OpenCV Show', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                print(f"Recording {wv_path}...")
                for frame_num in range(sequence_length):
                    ret, frame = cap.read() #đọc cap để lấy frame cho phần detect keypoints
                    if not ret:
                        print("Error reading camera.")
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    cv2.putText(image, f"Frame: {frame_num+1}/{sequence_length}", (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_TRAIN_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    VDWT.write(frame)
                    cv2.imshow('OpenCV Record', image)
                    cv2.waitKey(10)

                print(f"Recording completed: {wv_path}")
                VDWT.release()

    cap.release()
    cv2.destroyAllWindows()