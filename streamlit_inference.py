import io
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from typing import Any
from ultralytics.utils import LOGGER

# Cấu hình RTC (STUN và TURN servers)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:numb.viagenie.ca"],
                "username": "efIISG9C5KEPSFXKLR",
                "credential": "cDotpQhPLjS2EUPB",
            },
        ]
    }
)

class VideoTransformer(VideoTransformerBase):
    """Class to process video frames."""
    
    def __init__(self, model: YOLO, conf: float, iou: float, selected_ind: list):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.selected_ind = selected_ind
    
    def transform(self, frame: Any):
        """Transform the video frame using the YOLO model."""
        # Chuyển đổi frame từ BGR sang RGB để YOLO có thể xử lý
        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
        annotated_frame = results[0].plot()  # Vẽ các kết quả lên frame
        return annotated_frame

class Inference:
    """
    Class để thực hiện inference với YOLO và Streamlit WebRTC.
    """
    
    def __init__(self, model_path: str = None, conf: float = 0.25, iou: float = 0.45):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.model = None
        self.selected_ind = []

    def configure_model(self):
        """Tải model YOLO."""
        self.model = YOLO(self.model_path)  # Load model YOLO
        class_names = list(self.model.names.values())  # Danh sách tên lớp
        return class_names

    def run(self):
        """Chạy Streamlit ứng dụng với WebRTC."""
        # Cấu hình giao diện Streamlit
        self.web_ui()

        # Cấu hình Sidebar
        self.sidebar()

        # Cấu hình model và lớp chọn
        class_names = self.configure_model()
        
        selected_classes = st.sidebar.multiselect("Select Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]
        
        # Tạo và khởi chạy webrtc stream
        webrtc_streamer(
            key="unique_key_for_this_stream",
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=lambda: VideoTransformer(self.model, self.conf, self.iou, self.selected_ind),
        )

    def web_ui(self):
        """Thiết lập giao diện web cho Streamlit."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        # Thiết lập cấu hình cho trang web
        st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        st.markdown(menu_style_cfg, unsafe_allow_html=True)
        st.markdown(main_title_cfg, unsafe_allow_html=True)
        st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Thiết lập thanh bên cho Streamlit để chọn các tham số."""
        st.sidebar.title("User Configuration")
        self.model_path = st.sidebar.text_input("Model Path", "yolov8n.pt")  # Đường dẫn tới model YOLO
        self.conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))


if __name__ == "__main__":
    # Khởi tạo và chạy ứng dụng
    inf = Inference(model_path="yolov8n.pt")
    inf.run()
