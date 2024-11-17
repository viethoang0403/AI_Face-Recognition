from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import threading
import subprocess

app = Flask(__name__)

# Đường dẫn thư mục dữ liệu và file huấn luyện
DATASET_DIR = "datasets"
TRAINER_FILE = "Trainer.yml"

def run_datacollect(user_id):
    """Chạy datacollect.py để thu thập dữ liệu khuôn mặt."""
    try:
        subprocess.run(["python", "datacollect.py"], input=str(user_id), text=True)
    except Exception as e:
        print(f"Error running datacollect.py: {e}")


def run_training():
    """Chạy trainingdemo.py để huấn luyện mô hình."""
    try:
        subprocess.run(["python", "trainingdemo.py"])
    except Exception as e:
        print(f"Error running trainingdemo.py: {e}")

def run_stream():
    """Chạy testmodel.py để phát video trực tiếp."""
    try:
        subprocess.run(["python", "testmodel.py"])
    except Exception as e:
        print(f"Error running testmodel.py: {e}")

@app.route('/')
def home():
    """Trang chính."""
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register_face():
    """API Đăng ký khuôn mặt."""
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"status": "error", "message": "User ID is required!"}), 400

    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"status": "error", "message": "User ID must be an integer!"}), 400

    # Chạy datacollect.py trong luồng riêng
    thread = threading.Thread(target=run_datacollect, args=(user_id,))
    thread.start()

    return jsonify({"status": "success", "message": f"Data collection started for User ID {user_id}."})


@app.route('/train', methods=['POST'])
def train_model():
    """API Huấn luyện mô hình."""
    # Chạy trainingdemo.py trong luồng riêng
    thread = threading.Thread(target=run_training)
    thread.start()

    return jsonify({"status": "success", "message": "Training started."})

@app.route('/delete_faces', methods=['POST'])
def delete_faces():
    """API xóa dữ liệu khuôn mặt."""
    for f in os.listdir(DATASET_DIR):
        file_path = os.path.join(DATASET_DIR, f)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    # Xóa file Trainer.yml nếu tồn tại
    if os.path.exists(TRAINER_FILE):
        os.remove(TRAINER_FILE)

    return jsonify({"status": "success", "message": "All face data deleted."})

@app.route('/stream', methods=['POST'])
def stream():
    """API Phát trực tiếp."""
    # Chạy testmodel.py trong luồng riêng
    thread = threading.Thread(target=run_stream)
    thread.start()

    return jsonify({"status": "success", "message": "Streaming started."})

# # Danh sách tên cho nhận diện khuôn mặt
# name_list = ["", "son tung", "camila"]
#
# # Tải bộ phân loại Haar Cascade
# facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#
# # Kiểm tra và tạo LBPH Face Recognizer
# if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
# else:
#     raise Exception("Must import opencv-contrib-python to use LBPHFaceRecognizer.")
#
# # Biến kiểm soát việc stream video
# is_streaming = False
#
# def generate_frames():
#     """Tạo luồng video từ camera."""
#     video = cv2.VideoCapture(0)
#
#     # Đọc tệp huấn luyện
#     if os.path.exists("Trainer.yml"):
#         recognizer.read("Trainer.yml")
#     # else:
#     #     raise Exception("Trainer.yml not found.")
#
#     while True:
#         if is_streaming:  # Kiểm tra nếu flag stream đang bật
#             ret, frame = video.read()
#             if not ret:
#                 print("Unable to access Camera.")
#                 break
#             frame = cv2.flip(frame, 1)
#
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = facedetect.detectMultiScale(gray, 1.3, 5)
#
#             for (x, y, w, h) in faces:
#                 serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
#                 if conf > 50:
#                     name = name_list[serial] if serial < len(name_list) else "Unknown"
#                     color = (0, 255, 0)  # Green
#                 else:
#                     name = "Unknown"
#                     color = (0, 0, 255)  # Red
#
#                 # Vẽ khung và tên trên ảnh
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
#                 cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#
#             # Mã hóa khung hình thành JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#
#             # Gửi dữ liệu hình ảnh dưới dạng MJPEG stream
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         else:
#             # Nếu flag không bật, không làm gì cả
#             pass
#
#     # Giải phóng tài nguyên
#     video.release()
#     cv2.destroyAllWindows()
#
# @app.route('/video_feed')
# def video_feed():
#     """API cung cấp luồng video."""
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# @app.route('/start_stream', methods=['POST'])
# def start_stream():
#     """Bắt đầu phát trực tiếp."""
#     global is_streaming
#     is_streaming = True  # Bật flag stream
#     return jsonify({"status": "success", "message": "Streaming started. View the stream on the page."})
# @app.route('/stop_stream', methods=['POST'])
# def stop_stream():
#     """Dừng phát trực tiếp."""
#     global is_streaming
#     is_streaming = False  # Tắt flag stream
#     return jsonify({"status": "success", "message": "Streaming stopped."})

if __name__ == '__main__':
    app.run(debug=True)