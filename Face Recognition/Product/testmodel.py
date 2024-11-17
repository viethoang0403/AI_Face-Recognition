import cv2
import os
import time

# Khởi động camera
video = cv2.VideoCapture(0)

# Tải bộ phân loại Haar Cascade
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Kiểm tra và tạo LBPH Face Recognizer
if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    raise Exception("Must import opencv-contrib-python to use LBPHFaceRecognizer.")

# Đọc tệp huấn luyện
if os.path.exists("Trainer.yml"):
    recognizer.read("Trainer.yml")
else:
    raise Exception("Trainer.yml not found.")

background_path = "static/images/background.png"
if os.path.exists(background_path):
    imgBackground = cv2.imread(background_path)
else:
    imgBackground = None

# Danh sách tên cho nhận diện khuôn mặt
name_list = ["", "son tung", "camila"]

# Vòng lặp chính
while True:
    ret, frame = video.read()
    if not ret:
        print("Unable to access Camera.")
        break
    # frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf > 50:
            name = name_list[serial] if serial < len(name_list) else "Unknown"
            color = (0, 255, 0)  # Green
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red

        # Vẽ khung và tên trên ảnh
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Hiển thị khung hình trong nền (nếu có)
    if imgBackground is not None:
        resized_frame = cv2.resize(frame, (640, 480))
        imgBackground[162:162 + 480, 55:55 + 640] = resized_frame
        cv2.imshow("Stream", imgBackground)
    else:
        cv2.imshow("Stream", frame)

    # Thoát khi nhấn 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Giải phóng tài nguyên
video.release()
cv2.destroyAllWindows()
