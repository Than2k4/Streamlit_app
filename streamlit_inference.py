import streamlit as st
import cv2
import io
import base64
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

class Inference:
    def __init__(self, model_path=None):
        # Khởi tạo mô hình và các tham số cấu hình
        self.model = None
        self.model_path = model_path
        self.conf = 0.25  # Confidence threshold
        self.iou = 0.45  # IoU threshold
        self.selected_ind = []

        if self.model_path:
            self.model = YOLO(self.model_path)  # Tải mô hình YOLO nếu có đường dẫn mô hình

    def web_ui(self):
        """Tạo giao diện web cho Streamlit."""
        st.title("Ứng Dụng Phát Hiện Đối Tượng Real-time với YOLO")

    def sidebar(self):
        """Tạo Sidebar cho các cấu hình như threshold, mô hình..."""
        self.conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        self.iou = st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01)
        
    def capture_webcam(self):
        """Mở webcam và thực hiện phát hiện đối tượng."""
        cap = cv2.VideoCapture(0)  # Mở webcam

        if not cap.isOpened():
            st.error("Không thể mở webcam.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể lấy khung hình từ webcam.")
                break

            # Chạy YOLO inference trên khung hình
            results = self.model(frame, conf=self.conf, iou=self.iou)

            # Vẽ các kết quả lên khung hình
            annotated_frame = results[0].plot()

            # Hiển thị khung hình gốc và khung hình đã annotate
            st.image(frame, channels="BGR", caption="Khung Hình Gốc")
            st.image(annotated_frame, channels="BGR", caption="Khung Hình Phát Hiện Đối Tượng")

            # Chuyển đổi frame đã annotate thành base64 để gửi đi (nếu cần)
            img_pil = Image.fromarray(annotated_frame)
            buffered = BytesIO()
            img_pil.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            # Bạn có thể gửi base64 image này qua WebSocket hoặc các phương thức khác (chưa triển khai ở đây)
            # socket.emit('receive_img', img_base64)
            
            # Dừng inference khi nhấn nút
            if st.button("Dừng Inference"):
                cap.release()  # Dừng webcam
                st.stop()  # Dừng ứng dụng Streamlit

        cap.release()  # Giải phóng webcam

    def inference(self):
        """Chạy ứng dụng phát hiện đối tượng."""
        self.web_ui()  # Giao diện chính
        self.sidebar()  # Sidebar để người dùng cấu hình

        # Bắt đầu capture webcam
        self.capture_webcam()

if __name__ == "__main__":
    # Khởi tạo và chạy ứng dụng Inference
    inference = Inference(model_path="yolov5s.pt")  # Thay "yolov5s.pt" bằng đường dẫn mô hình YOLO của bạn
    inference.inference()
